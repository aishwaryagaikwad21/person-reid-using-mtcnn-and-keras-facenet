# person-reid-using-mtcnn-and-keras-facenet
Person Reidentification using mtcnn for face detection and keras-facenet for recognition
- git clone https://github.com/aishwaryagaikwad21/person-reid-using-mtcnn-and-keras-facenet
- Download model from [link](https://drive.google.com/drive/folders/14UntmrJuCO9uTdzMKnLIvuDw2Wwkfwdn)
```
Run face_detect.py 
  - MTCNN will detect faces then a compressed numpy array file '.npz' will be created 
 ```
 ```
Run face_embeddings.py
  - Face embeddings -> vector that represents the features extracted from the face
  - It will first load numpy array values of detected faces
  - Facenet model will convert arrays to face embeddings
```
```
Run fittingLinearSvm.py 
  - Linear SVM will take face embeddings as input 
  - Evaluate the model
  - Fitting a linear SVM on face embeddings
```
```
Run person_reid.py
  - Predicting the identity for a given unseen photo in the test dataset
```
 
