import cv2 as cv
import numpy as np

import uvicorn
import aiofiles
from typing import List
from pydantic import BaseModel
from models import User, get_db
from sqlalchemy.orm import Session
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Body

from encrypt import encrypt_data
from face_recognition import FaceRecognition
from qdrant_client_wrapper import registration, search

STORAGE_DIR = "./storage/images"
fr = FaceRecognition()

app = FastAPI()


class RegistrationData(BaseModel):
    name: str


@app.post("/register/{name}")
async def register_user(
    name: str,
    images: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    if len(images) != 3:
        raise HTTPException(status_code=400, detail="Exactly 3 images are required.")

    image_paths = []

    ct_success = 0
    user_id = None
    for i, image in enumerate(images):
        filename = f"{name}_{i:02d}.jpg"
        image_path = f"{STORAGE_DIR}/{filename}"

        async with aiofiles.open(image_path, "wb") as f:
            while content := await image.read(1024):  # Read image in chunks
                await f.write(content)

        image_paths.append(image_path)

        img = cv.imread(image_path)
        ret, _, face_crop = fr.detect(img)
        if not ret:
            raise HTTPException(
                status_code=400, detail=f"No face detected in image {i+1}."
            )
        # cv.imwrite(image_path, face_crop)
        embedding = fr.get_embedding_vector(img)

        if user_id is None:
            db_user = User(name=name, image_paths=str(image_paths))
            db.add(db_user)
            db.commit()
            db.refresh(db_user)
            user_id = db_user.id
        else:
            db_user = db.query(User).filter(User.id == user_id).first()
            db_user.image_paths = str(image_paths)
            db.commit()

        is_success = registration(embedding, int(user_id))

        if is_success:
            ct_success += 1
        else:
            break

    if ct_success != 3:
        raise HTTPException(status_code=500, detail="Failed to register user.")

    response_data = {
        "message": "User registered successfully!",
        "name": name,
        "image_paths": image_paths,
        "user_id": user_id,
    }
    return {"data": encrypt_data(response_data)}


@app.post("/identify/")
async def identify_user(
    images: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    top_k = 3
    if len(images) != 1:
        raise HTTPException(
            status_code=400, detail="Exactly 1 image is required for identification."
        )

    image = images[0]
    filename = "query.jpg"
    image_path = f"{STORAGE_DIR}/{filename}"

    async with aiofiles.open(image_path, "wb") as f:
        while content := await image.read(1024):  # Read image in chunks
            await f.write(content)

    img = cv.imread(image_path)
    ret, face_norm, face_crop = fr.detect(img)
    if not ret:
        raise HTTPException(status_code=400, detail=f"No face detected in image.")
    # cv.imwrite(image_path, face_crop)
    threshold = fr.get_verify_threshold()
    query_embedding = fr.get_embedding_vector(img)

    distances, indices = search(query_embedding, k=top_k)
    distances = 1 - np.array(distances)  # Convert cosine similarity to cosine distance
    candidate_ids = [
        int(idx) for dist, idx in zip(distances, indices) if dist < threshold
    ]
    if not candidate_ids:
        raise HTTPException(status_code=404, detail="No matching faces found.")

    id_count = {}
    for candidate_id in candidate_ids:
        id_count[candidate_id] = id_count.get(candidate_id, 0) + 1

    matched_user_id = None
    matched_score = None
    for user_id, count in id_count.items():
        if count >= 2:
            matched_user_id = user_id
            matched_score = float(abs(np.round(np.min(distances), 4)))
            break

    if matched_user_id is None:
        raise HTTPException(
            status_code=404, detail="No sufficient matches found for identification."
        )

    user = db.query(User).filter(User.id == matched_user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found in database.")

    response_data = {
        "message": "User identified successfully!",
        "user_id": user.id,
        "name": user.name,
        "image_paths": user.image_paths,
        "score": matched_score,
    }
    return {"data": encrypt_data(response_data)}


@app.post("/verify/")
async def verify_user(
    images: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    if len(images) != 2:
        raise HTTPException(
            status_code=400, detail="Exactly 2 images are required for verification."
        )

    image_paths = []
    for i, image in enumerate(images):
        filename = f"verify_{i:02d}.jpg"
        image_path = f"{STORAGE_DIR}/{filename}"

        async with aiofiles.open(image_path, "wb") as f:
            while content := await image.read(1024):  # Read image in chunks
                await f.write(content)

        image_paths.append(image_path)

    img1 = cv.imread(image_paths[0])
    img2 = cv.imread(image_paths[1])

    ret, _, face_crop = fr.detect(img1)
    if not ret:
        raise HTTPException(status_code=400, detail=f"No face detected in image 1.")

    ret, _, face_crop = fr.detect(img2)
    if not ret:
        raise HTTPException(status_code=400, detail=f"No face detected in image 2.")

    result, distance = fr.verify(img1, img2)

    if result:
        response_data = {
            "message": "Faces match!",
            "distance": distance,
        }
        return {"data": encrypt_data(response_data)}
    else:
        raise HTTPException(
            status_code=400, detail=f"Faces do not match. Distance: {distance}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
