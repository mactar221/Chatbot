import streamlit as st
from PIL import Image, ImageDraw
from deepface import DeepFace
import tempfile
import os

# Fonction pour détecter les visages
def detect_faces(image):
    # Convertir l'image PIL en fichier temporaire
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        image.save(tmp_file, format='JPEG')
        tmp_file_path = tmp_file.name
    
    # Utiliser deepface pour détecter les visages
    analysis = DeepFace.analyze(img_path=tmp_file_path, actions=['face_detection'])
    
    # Supprimer le fichier temporaire
    os.remove(tmp_file_path)
    
    face_boxes = analysis[0]['region']
    return face_boxes

# Fonction pour dessiner et enregistrer l'image avec les visages détectés
def draw_faces(image, face_boxes):
    draw = ImageDraw.Draw(image)
    for face_box in face_boxes:
        x, y, w, h = face_box['x'], face_box['y'], face_box['w'], face_box['h']
        draw.rectangle([x, y, x + w, y + h], outline="red", width=4)
    return image

# Interface utilisateur avec Streamlit
def main():
    st.title('Détection de visages avec PIL et DeepFace')

    uploaded_file = st.file_uploader("Uploader une image", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        # Lire le fichier image avec PIL
        image = Image.open(uploaded_file)

        st.image(image, caption='Image originale')

        if st.button('Détecter les visages'):
            face_boxes = detect_faces(image)
            image_with_faces = draw_faces(image, face_boxes)
            st.image(image_with_faces, caption='Image avec visages détectés')

            # Save the image with detected faces
            output_path = 'image_with_faces.jpg'  # Nom du fichier de sortie
            image_with_faces.save(output_path)

            with open(output_path, 'rb') as file:
                st.download_button(
                    label="Télécharger image avec visages détectés",
                    data=file,
                    file_name="image_with_faces.jpg",
                    mime="image/jpeg"
                )

if __name__ == '__main__':
    main()



