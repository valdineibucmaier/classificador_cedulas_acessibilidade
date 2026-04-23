import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps
from gtts import gTTS
import base64
import os
import time
import numpy as np
import cv2




# Configuração da página
st.set_page_config(page_title="Classificador cédulas - Acessibilidade")

# --- ESTILO CSS PARA ACESSIBILIDADE ---
st.markdown("""
    <style>
    
    /* Estiliza a área da câmera para ser mais visível */
    div[data-testid="stCameraInput"] {
        border: 4px dashed #007bff !important;
        border-radius: 15px !important;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- FUNCAO TRANSFORMA EM ESCALA DE CINZA ---
def transformar_em_cinza(image):
    """
    Converte a imagem para escala de cinza e mantém os 3 canais RGB.
    """
    # 1. Converte para escala de cinza (L)
    grayscale_img = ImageOps.grayscale(image)
    
    # 2. Converte de volta para RGB
    # Isso é necessário porque o modelo espera uma entrada de 3 canais (R, G, B),
    # mesmo que todos tenham a mesma informação de cinza.
    return grayscale_img.convert("RGB")

# --- FUNCAO DE CORRECAO DA COR SEM LUZ NATURAL
def corrigir_balanco_branco(img_pil):
    # Converte PIL para OpenCV (formato que processa cores melhor)
    img = np.array(img_pil)
    
    # Calcula a média de cada canal
    avg_r = np.average(img[:, :, 0])
    avg_g = np.average(img[:, :, 1])
    avg_b = np.average(img[:, :, 2])
    
    # Calcula a média cinza (teoria do mundo cinza)
    avg_gray = (avg_r + avg_g + avg_b) / 3
    
    # Ajusta os canais para neutralizar o amarelado
    img[:, :, 0] = np.minimum(img[:, :, 0] * (avg_gray / avg_r), 255)
    img[:, :, 1] = np.minimum(img[:, :, 1] * (avg_gray / avg_g), 255)
    img[:, :, 2] = np.minimum(img[:, :, 2] * (avg_gray / avg_b), 255)
    
    return Image.fromarray(img.astype('uint8'))

# --- FUNCAO ROBUSTA CORRECAO IMAGEM ---
def tratar_imagem_robusta(img_pil):
    # Converte PIL para OpenCV (BGR)
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR).astype(np.float32)
    
    # --- BALANÇO DE BRANCO MANUAL (Grey World) ---
    # Calcula a média de cada canal
    avg_b = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_r = np.mean(img[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3
    
    # Ajusta os canais (evita divisão por zero)
    img[:, :, 0] *= (avg_gray / (avg_b + 1e-6))
    img[:, :, 1] *= (avg_gray / (avg_g + 1e-6))
    img[:, :, 2] *= (avg_gray / (avg_r + 1e-6))
    
    # Clipar valores entre 0-255 e voltar para uint8
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    # --- MELHORIA DE CONTRASTE (CLAHE) ---
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# --- FUNÇÃO DE ÁUDIO ---
def falar(texto):
    tts = gTTS(text=texto, lang='pt-BR')
    filename = "temp_audio.mp3"
    tts.save(filename)
    
    with open(filename, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        # O 'autoplay' faz a voz sair assim que a classificação termina
        audio_html = f"""
            <audio autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
    os.remove(filename) # Limpa o arquivo temporário

# --- FUNÇÕES DE IA ---

@st.cache_resource
def load_model():
    # 1. Instancia a MobileNetV2
    model = models.mobilenet_v2(weights=None)
    
    # 2. Ajuste a última camada (Linear) para o seu número de classes
    # Se você tem 7 cédulas (2, 5, 10, 20, 50, 100, 200)
    n_inputs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(n_inputs, 8) 
    
    # 3. Carregar os pesos
    model.load_state_dict(torch.load('melhor_modelo_otimizado_decisivo_epoca_13.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(image, model):

    #imagem_cinza = transformar_em_cinza(image)
    #imagem_corrigida = corrigir_balanco_branco(image)
    #imagem_corrigida = tratar_imagem_robusta(image)
    # Transformações (devem ser IGUAIS às do treinamento no Colab)
    
    # preprocess = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])


    
     
    
    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        # 1.2 de contraste e 0.8 de saturação (fixos)
        transforms.ColorJitter(contrast=1.2, saturation=0.8),
        # O ColorJitter aqui vai 'estressar' a imagem para neutralizar o vício da câmera
        #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


   
    
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0) # Cria o "lote" de 1 imagem

    with torch.no_grad():
        output = model(input_batch)
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # Substitua pela sua lista de classes na ordem correta
    classes = ['nota-10', 'nota-100', 'nota-2', 'nota-20', 'nota-200', 'nota-5', 'nota-50', 'outros']
    
    conf, idx = torch.max(probabilities, 0)
    return classes[idx], conf.item()

# Mapeamento técnico para humano
NOMES_AMIGAVEIS = {
    'nota-2': '2 reais',
    'nota-5': '5 reais',
    'nota-10': '10 reais',
    'nota-20': '20 reais',
    'nota-50': '50 reais',
    'nota-100': '100 reais',
    'nota-200': '200 reais',
    'outros': 'objeto não identificado'
}

# --- INICIALIZAÇÃO DO ESTADO ---
if 'camera_key' not in st.session_state:
    st.session_state.camera_key = 0

# --- ESTILO CSS ---
st.markdown("""
    <style>
    /* Estiliza apenas o botão nativo da câmera para ser gigante */
    div[data-testid="stCameraInput"] button {
        width: 100% !important;
        height: 120px !important;
        font-size: 25px !important;
        background-color: #28a745 !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Identificador de Cédulas v.Prime")

# --- AVISO INICIAL DE USO ---
# Usamos o session_state para que o áudio de boas-vindas toque apenas UMA vez ao abrir
if 'avisado' not in st.session_state:
    aviso_texto = "Bem-vindo! Para uma leitura precisa, fotografe uma cédula por vez."
    st.info(aviso_texto)
    falar(aviso_texto)
    st.session_state.avisado = True

# A key muda toda vez que queremos dar 'Clear' na foto
camera_file = st.camera_input("TOQUE PARA TIRAR FOTO", key=f"cam_{st.session_state.camera_key}")

if camera_file:
    # 1. Processamento Automático (Sem botão extra!)
    image = Image.open(camera_file).convert('RGB')
    
    with st.spinner('Identificando...'):
        model = load_model()
        label, confidence = predict(image, model)
        
        if confidence > 0.70:
            nome_fala = NOMES_AMIGAVEIS.get(label, label)
            if label == 'outros':
                res = "Isso não parece ser uma nota de Real."
                st.warning(res)
                falar(res)

                time.sleep(4) # Tempo para ouvir o áudio
                st.session_state.camera_key += 1 # Muda a key, o que "mata" a foto anterior
                st.rerun() # Reinicia o app já com a câmera limpa
            else:
                res = f"Nota de {nome_fala} identificada."
                st.success(res)
                falar(res)
                # No sucesso, deixamos a foto na tela para confirmação visual
        else:
            erro = "Não identifiquei. Tente novamente."
            st.warning(erro)
            falar(erro)
            
            # 2. O SEGREDO PARA O CLEAR AUTOMÁTICO:
            time.sleep(4) # Tempo para ouvir o áudio
            st.session_state.camera_key += 1 # Muda a key, o que "mata" a foto anterior
            st.rerun() # Reinicia o app já com a câmera limpa
        st.metric("Confiança da IA", f"{confidence*100:.2f}%")