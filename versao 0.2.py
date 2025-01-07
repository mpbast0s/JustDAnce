#!/usr/bin/env python
# coding: utf-8

# In[62]:


print('oi')


# In[63]:


##


# In[15]:


get_ipython().system('pip install tensorflow')


# In[16]:


get_ipython().system('pip install opencv-python matplotlib')


# In[17]:


get_ipython().system('pip install opencv-python')


# In[2]:


import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2


# In[3]:


interpreter = tf.lite.Interpreter(model_path='lite-model_movenet_singlepose_lightning_3.tflite')
interpreter.allocate_tensors()


# In[4]:


####


# In[5]:


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0,255,0), -1) 


# In[6]:


EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}
def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)


# In[7]:


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    # Reshape image
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192,192)
    input_image = tf.cast(img, dtype=tf.float32)
    
    # Setup input and output 
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Make predictions 
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    
    # Rendering 
    draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
    draw_keypoints(frame, keypoints_with_scores, 0.4)
    
    cv2.imshow('MoveNet Lightning', frame)
    
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()


# In[9]:


print(keypoints_with_scores)


# In[25]:


import cv2
import time

cap = cv2.VideoCapture(0) # Abre a câmera padrão do computador
fps = 30 # FPS desejado

while True:
    # Marca o tempo inicial
    start_time = cv2.getTickCount() 
    start_tm = time.time() 

    # Código para capturar e processar o frame aqui
    ret, frame = cap.read()
     
    # Calcula o tempo de espera necessário para atingir o FPS desejado
    time_per_frame = cv2.getTickCount() - start_time
    time_per_frame /= cv2.getTickFrequency()
    wait_time = max(1, int((1000/fps) - time_per_frame))
    
    #ainda não está como eu queriaaaaa
    #####
    fpss = int(1.0 / (time.time() - start_tm))
    fps_text = "FPS: " + str(fpss)
    
    cv2.putText(frame,fps_text,(10,30),cv2.FONT_HERSHEY_PLAIN,1,255)
    
    cv2.imshow('JustDance 2.1', frame)

    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[26]:


fps


# In[27]:


import cv2
import time

cap = cv2.VideoCapture(0) # Abre a câmera padrão do computador
fps = 30 # FPS desejado

while cap.isOpened():
    ret, frame = cap.read()
    
     # Marca o tempo inicial
    start_time = cv2.getTickCount() 
    start_tm = time.time() 
    
    # Reshape image
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192,192)
    input_image = tf.cast(img, dtype=tf.float32)
    
    # Setup input and output 
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Make predictions 
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    
    # Rendering 
   # draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
    draw_keypoints(frame, keypoints_with_scores, 0.4)
    
    
    # Calcula o tempo de espera necessário para atingir o FPS desejado
    time_per_frame = cv2.getTickCount() - start_time
    time_per_frame /= cv2.getTickFrequency()
    wait_time = max(1, int((1000/fps) - time_per_frame))
    
    #ainda não está como eu queriaaaaa
    #####
    fpss = int(1.0 / (time.time() - start_tm))
    fps_text = "FPS: " + str(fpss)
    
    cv2.putText(frame,fps_text,(10,30),cv2.FONT_HERSHEY_PLAIN,1,255)
    
    cv2.imshow('JustDance 2.1', frame)

    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()


# In[10]:


import cv2
import time

cap = cv2.VideoCapture("meuvideo.mp4") # Abre a câmera padrão do computador
fps = 30 # FPS desejado

while cap.isOpened():
    ret, frame = cap.read()
    
     # Marca o tempo inicial
    start_time = cv2.getTickCount() 
    start_tm = time.time() 
    
    # Reshape image
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192,192)
    input_image = tf.cast(img, dtype=tf.float32)
    
    # Setup input and output 
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Make predictions 
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    
    # Rendering 
   # draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
    draw_keypoints(frame, keypoints_with_scores, 0.4)
    
    
    # Calcula o tempo de espera necessário para atingir o FPS desejado
    time_per_frame = cv2.getTickCount() - start_time
    time_per_frame /= cv2.getTickFrequency()
    wait_time = max(1, int((1000/fps) - time_per_frame))
    
    #ainda não está como eu queriaaaaa
    #####
    fpss = int(1.0 / (time.time() - start_tm))
    fps_text = "FPS: " + str(fpss)
    
    cv2.putText(frame,fps_text,(10,30),cv2.FONT_HERSHEY_PLAIN,1,255)
    
    cv2.imshow('JustDance 2.1', frame)

    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()


# In[29]:


get_ipython().system('pip install pytube')


# In[30]:


from pytube import YouTube

def baixar_video(link, titulovideo):
    # URL do vídeo do YouTube
    # url = "https://www.youtube.com/watch?v=jPa748Wvo6M&ab_channel=CaaSilva"

    # Nome desejado para o arquivo de saída
    output_filename = titulovideo

    # Criar um objeto da classe YouTube
    yt = YouTube(link)

    # Selecionar a melhor qualidade disponível
    ys = yt.streams.filter(progressive=True, res="360p").first()
    #stream = video.streams.filter(progressive=True, res="360p").first()


    # Fazer o download do vídeo com o nome desejado
    #ys.download(output_path='./', filename=output_filename)
    ys.download(filename=output_filename)
    return

titulo = input("Nome do arquivo: ")
titulovideo = titulo + ".mp4"
link = input("Digite o link do vídeo do YouTube: ")
baixar_video(link, titulovideo)


# In[ ]:





# In[31]:


import cv2
import time
import csv
filename = titulo + ".csv"
output_images = []
numero_frames = 0

cap = cv2.VideoCapture(titulovideo) # Abre a câmera padrão do computador
#fps = 30 # FPS desejado

with open(filename, 'w') as csvfile:
    while cap.isOpened():
        ret, frame = cap.read()
        # Check if frame was read successfully
        if not ret:
            break

         # Marca o tempo inicial
        start_time = cv2.getTickCount() 
        start_tm = time.time() 

        # Reshape image
        img = frame.copy()
        img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192,192)
        input_image = tf.cast(img, dtype=tf.float32)
        

        # Setup input and output 
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Make predictions 
        interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
        interpreter.invoke()
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

        # Rendering 
        draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
        draw_keypoints(frame, keypoints_with_scores, 0.4)

        #pegar keypoints:
        keypoints_with_scores_new = []
        for sample in keypoints_with_scores:
          new_sample = []
          for person in sample:
              new_person = []
              for index, keypoint in enumerate(person):
                  # Adicionar o índice antes de cada trio
                  new_person.append([index] + keypoint.tolist())
              new_sample.append(new_person)
          keypoints_with_scores_new.append(new_sample)

          indices_desejados = [0] + list(range(5, 17))
          csvwriter = csv.writer(csvfile)
          for row in keypoints_with_scores_new:
            for row in row:
              for row in row:
                  if row[0] in indices_desejados:
                    csvwriter.writerow(row)


        #ainda não está como eu queriaaaaa
        #####
        #fpss = int(1.0 / (time.time() - start_tm))
        #fps_text = "FPS: " + str(fpss)

        #cv2.putText(frame,fps_text,(10,30),cv2.FONT_HERSHEY_PLAIN,1,255)
        
        numero_frames = numero_frames + 1
        
        cv2.imshow('JustDance 2.3', frame)

        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


# In[32]:


get_ipython().system('pip install moviepy')


# In[11]:


from moviepy.video.io.VideoFileClip import VideoFileClip

# Carregar o arquivo de vídeo
video = VideoFileClip(titulovideo)

# Obter a duração do vídeo em segundos
duracao = video.duration

# Imprimir a duração do vídeo em segundos
print("Duração do vídeo:", duracao, "segundos")

# Fechar o arquivo de vídeo
video.close()


# In[34]:


numero_frames


# In[35]:


fps = numero_frames / duracao
print(fps)


# In[15]:


import math

def angle_between_points(a, b, c):
    """
    Calculates the angle between three points (a, b, c) represented as (x, y) coordinates.
    The angle is calculated at point b.
    a = [x, y, score]
    b = []
    """
    # Calculate the vectors ab and bc
    ab = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    
    # Calculate the dot product and the magnitude of each vector
    dot_product = ab[0]*bc[0] + ab[1]*bc[1]
    magnitude_ab = math.sqrt(ab[0]**2 + ab[1]**2)
    magnitude_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    
    # Calculate the cosine of the angle using the dot product and magnitudes
    cos_angle = dot_product / (magnitude_ab * magnitude_bc)
    
    # Calculate the angle in radians and convert it to degrees
    angle_in_radians = math.acos(cos_angle)
    angle_in_degrees = math.degrees(angle_in_radians)
    
    #calculo da media dos scores:
    media_1 =( a[2] +  b[2] + c[2])/3 
    
    return [ angle_in_degrees , media_1 ]


# In[16]:


import cv2
import time
import csv
titulo = "meuvideo"
titulovideo = titulo + ".mp4"
filename = titulo + "angulos.csv"
output_images = []
numero_frames = 0

cap = cv2.VideoCapture(titulovideo) # Abre a câmera padrão do computador
#fps = 30 # FPS desejado

with open(filename, 'w') as csvfile:
    while cap.isOpened():
        ret, frame = cap.read()
        # Check if frame was read successfully
        if not ret:
            break

         # Marca o tempo inicial
        start_time = cv2.getTickCount() 
        start_tm = time.time() 

        # Reshape image
        img = frame.copy()
        img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192,192)
        input_image = tf.cast(img, dtype=tf.float32)
        

        # Setup input and output 
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Make predictions 
        interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
        interpreter.invoke()
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

        # Rendering 
        draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
        draw_keypoints(frame, keypoints_with_scores, 0.4)

        #pegar keypoints:
        keypoints_with_scores_new = []
        for sample in keypoints_with_scores:
          new_sample = []
          for person in sample:
              new_person = []
              for index, keypoint in enumerate(person):
                  # Adicionar o índice antes de cada trio
                  new_person.append([index] + keypoint.tolist())
              new_sample.append(new_person)
          keypoints_with_scores_new.append(new_sample)

          indices_desejados = [0] + list(range(5, 17))
          csvwriter = csv.writer(csvfile)
          keypoints_dict = {}
          for row in keypoints_with_scores_new:
            for row in row:
              for row in row:
                  if row[0] in indices_desejados:
                   keypoints_dict[row[0]] = [row[1], row[2], row[3]]

          key_0 = keypoints_dict.get(0)
          key_5 = keypoints_dict.get(5)
          key_6 = keypoints_dict.get(6)
          key_7 = keypoints_dict.get(7)
          key_8 = keypoints_dict.get(8)
          key_9 = keypoints_dict.get(9)
          key_10 = keypoints_dict.get(10)
          key_11 = keypoints_dict.get(11)
          key_12 = keypoints_dict.get(12)
          key_13 = keypoints_dict.get(13)
          key_14 = keypoints_dict.get(14)
          key_15 = keypoints_dict.get(15)
          key_16 = keypoints_dict.get(16)
        
                     
        # Lista com as tuplas dos pontos para calcular os ângulos
        pontos = [(key_5, key_0, key_6), 
                  (key_5, key_7, key_9), (key_6, key_8, key_10),
                  (key_9, key_11, key_13), (key_10, key_12, key_14),
                  (key_11, key_13, key_15), (key_12, key_14, key_16),]
        
        # Lista para armazenar os valores de ângulo
        angulos = []

        # Calcular e armazenar os ângulos
        for p in pontos:
            angulo = angle_between_points(*p)
            angulos.append([angulo])

        # Escrever os valores de ângulo no arquivo
        for row in angulos:
            for row in row:
                csvwriter.writerow(row)



        #ainda não está como eu queriaaaaa
        #####
        #fpss = int(1.0 / (time.time() - start_tm))
        #fps_text = "FPS: " + str(fpss)

        #cv2.putText(frame,fps_text,(10,30),cv2.FONT_HERSHEY_PLAIN,1,255)
        
        numero_frames = numero_frames + 1
        
        cv2.imshow('JustDance 2.4', frame)

        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


# In[99]:


key_0


# In[101]:


keypoints_dict


# In[109]:


med_1


# In[11]:


angulos


# In[4]:


import cv2

# Inicia captura de vídeo
cap = cv2.VideoCapture('meuvideo.mp4')

# Inicia captura de imagem da câmera
cam = cv2.VideoCapture(0)

while True:
    # Lê frame do vídeo
    ret, frame = cap.read()

    # Lê frame da câmera
    ret_cam, frame_cam = cam.read()

    frame = cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))

    # Corta a imagem da câmera para o mesmo tamanho do vídeo
    # Corta a imagem da câmera centralmente
    frame_cam = frame_cam[frame.shape[1]:frame_cam.shape[1],frame.shape[0]:frame_cam.shape[0]]
    
    # Redimensiona a imagem da câmera para que tenha o mesmo tamanho do vídeo
    frame_cam = cv2.resize(frame_cam, (frame.shape[1], frame.shape[0]))

    # Concatena as imagens horizontalmente
    combined_frame = cv2.hconcat([frame, frame_cam])

    # Exibe a imagem combinada em uma janela
    cv2.imshow('Combined Video and Camera Feed', combined_frame)

    # Aguarda 1ms por uma tecla para ser pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Verifica se o vídeo terminou
    if not cap.isOpened():
        break

# Libera os recursos
cap.release()
cam.release()
cv2.destroyAllWindows()


# In[ ]:





# In[20]:


import cv2
import tkinter as tk

# Função a ser chamada quando o botão "Começar" for pressionado
def iniciar_video():
    # Inicia captura de vídeo
    cap = cv2.VideoCapture('meuvideo.mp4')

    # Inicia captura de imagem da câmera
    cam = cv2.VideoCapture(0)

    while True:
        # Lê frame do vídeo
        ret, frame = cap.read()

        # Lê frame da câmera
        ret_cam, frame_cam = cam.read()

        frame = cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))

        # Corta a imagem da câmera para o mesmo tamanho do vídeo
        # Corta a imagem da câmera centralmente
        frame_cam = frame_cam[frame.shape[1]:frame_cam.shape[1],frame.shape[0]:frame_cam.shape[0]]

        # Redimensiona a imagem da câmera para que tenha o mesmo tamanho do vídeo
        frame_cam = cv2.resize(frame_cam, (frame.shape[1], frame.shape[0]))

        # Concatena as imagens horizontalmente
        combined_frame = cv2.hconcat([frame, frame_cam])

        # Exibe a imagem combinada em uma janela
        cv2.imshow('Combined Video and Camera Feed', combined_frame)

        # Aguarda 1ms por uma tecla para ser pressionada
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Verifica se o vídeo terminou
        if not cap.isOpened():
            break

    # Libera os recursos
    cap.release()
    cam.release()
    cv2.destroyAllWindows()



# Define as configurações da janela
janela = tk.Tk()
janela.geometry('300x150')
janela.title('Janela de Início')

# Define o título e a fonte do botão "Começar"
titulo = tk.Label(janela, text='Bem-vindo(a)!', font=('Arial', 16))
titulo.pack(pady=10)
botao = tk.Button(janela, text='Começar', font=('Arial', 14), command=iniciar_video, bg='#000')
botao.pack(fill='both', expand=True, padx=20, pady=10)

# Define a cor de fundo da janela
janela.configure(bg='#bbb')

# Inicia a janela
janela.mainloop()


# In[17]:


import cv2
import tkinter as tk
import time

def iniciar_video():
    janela.destroy()
    
    cap = cv2.VideoCapture(0)

    # Inicializa a variável "frame"
    frame = None
    
    # Define o momento de início do jogo
    inicio_jogo = None
    
    # Define o tempo máximo para o jogo (em segundos)
    tempo_maximo_jogo = 30
    
    while True:
        # Lê frame da câmera
        ret, frame = cap.read()
        
        # Verifica se a câmera foi lida corretamente
        if not ret:
            break
        
        # Exibe a imagem em uma janela
        cv2.imshow('Camera Feed', frame)
        
        # Exibe o texto "Balance a mão para começar o jogo" por 5 segundos
        if inicio_jogo is None:
            cv2.putText(frame, "Balance a mão para começar o jogo", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Camera Feed', frame)
            cv2.waitKey(5000)
            inicio_jogo = time.time()

        # Verifica se o tempo máximo do jogo foi atingido
        tempo_decorrido = time.time() - inicio_jogo
        if tempo_decorrido >= tempo_maximo_jogo:
            break

        # Verifica se a tecla "q" foi pressionada
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera os recursos
    cap.release()
    cv2.destroyAllWindows()

# Cria janela com o botão "Começar"
janela = tk.Tk()
janela.geometry('200x100')
botao = tk.Button(janela, text='Começar', command=iniciar_video)
botao.pack(fill='both', expand=True)
janela.mainloop()


# In[19]:


import cv2

cap = cv2.VideoCapture(0)

# Define ROI para detectar a mão
x, y, w, h = 100, 100, 200, 200

# Inicializa o primeiro frame
ret, frame_ant = cap.read()
gray_ant = cv2.cvtColor(frame_ant, cv2.COLOR_BGR2GRAY)
gray_ant = cv2.GaussianBlur(gray_ant, (21, 21), 0)

while True:
    # Lê frame atual
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Calcula diferença entre frames
    diff = cv2.absdiff(gray_ant, gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Desenha ROI na imagem
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Verifica se a mão foi balançada
    roi = thresh[y:y+h, x:x+w]
    if cv2.countNonZero(roi) > 500:
        print("Balançou a mão!")

    # Atualiza o frame anterior
    gray_ant = gray

    # Exibe a imagem em uma janela
    cv2.imshow("Camera", frame)

    # Aguarda 1ms por uma tecla para ser pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




