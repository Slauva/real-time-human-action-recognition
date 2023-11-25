import timm
import pickle
from tqdm import tqdm
from torchvision import transforms
from vit_pytorch.vit_3d import ViT

import torch
from torch import nn
import numpy as np
import cv2
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
import tensorflow as tf

def evaluate(model, val_data, loss_fn, weights = None, device = 'cpu', verbose = 0):
    
    if (device == 'gpu' or device == 'cuda') and torch.cuda.is_available():
        device = torch.device('cuda')
    elif isinstance(device, torch.device): 
        device = device
    else: 
        device = torch.device('cpu')
    
    model = model.to(device)

    if weights:
        model.load_state_dict(torch.load(weights))
    
    with torch.no_grad():
        model.eval()
        val_correct = 0
        val_total = len(val_data)*val_data.batch_size
        running_loss = 0.
        if verbose == 1:
            val_data = tqdm(val_data, desc = 'Evaluate: ', ncols = 100)
        for data_batch, label_batch in val_data:
            
            data_batch, label_batch = data_batch.to(device), label_batch.to(device)
            output_batch = model(data_batch)
            loss = loss_fn(output_batch, label_batch.long())
            running_loss += loss.item()

            _, predicted_labels = torch.max(output_batch.data, dim = 1)

            val_correct += (label_batch == predicted_labels).sum().item()
        val_loss = running_loss/len(val_data)
        val_acc = val_correct/val_total
        return val_loss, val_acc


def train(model, train_data, val_data, loss_fn, optimizer, epochs, save_last_weights_path = None,
          save_best_weights_path = None, freeze = False, steps_per_epoch = None,
          device = 'cpu', scheduler = None):

    if (device == 'gpu' or device == 'cuda') and torch.cuda.is_available():
        device = torch.device('cuda')
    elif isinstance(device, torch.device): 
        device = device
    else: 
        device = torch.device('cpu')        

    if save_best_weights_path: 
        best_loss, _ = evaluate(model, val_data, device = device, loss_fn = loss_fn, verbose = 1)  

    if steps_per_epoch is None: 
        steps_per_epoch = len(train_data)

    num_steps = len(train_data)
    iterator = iter(train_data)
    count_steps = 1    
    
    ## History
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_loss': []
    }
        
    # add model to device
    model = model.to(device)
    
    ############################### Train and Val ##########################################
    for epoch in range(1, epochs + 1):

        running_loss = 0.
        train_correct = 0
        train_total = steps_per_epoch*train_data.batch_size
        

        model.train()
        
        for step in tqdm(range(steps_per_epoch), desc = f'epoch: {epoch}/{epochs}: ', ncols = 100): 
            
            img_batch, label_batch = next(iterator)
            img_batch, label_batch = img_batch.to(device), label_batch.to(device)
            

            optimizer.zero_grad()
            

            output_batch = model(img_batch)
            

            loss = loss_fn(output_batch, label_batch.long())
            

            loss.backward()
            

            optimizer.step()
            

            _, predicted_labels = torch.max(output_batch.data, dim = 1)

            train_correct += (label_batch == predicted_labels).sum().item()
                

            running_loss += loss.item()
                
            if count_steps == num_steps:
                count_steps = 0
                iterator = iter(train_data)
            count_steps += 1
            
        train_loss = running_loss / steps_per_epoch
        train_accuracy = train_correct/train_total
        
        if scheduler:
            scheduler.step(train_loss)
        
        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_accuracy))
        
        if val_data is not None: 
            val_loss, val_acc = evaluate(model, val_data, device = device, loss_fn = loss_fn)
            print(f'epoch: {epoch}, train_accuracy: {train_accuracy: .2f}, loss: {train_loss: .3f}, val_accuracy: {val_acc: .2f}, val_loss: {val_loss:.3f}')

            if save_best_weights_path:
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), save_best_weights_path)
                    print(f'Saved successfully best weights to:', save_best_weights_path)
            history['val_loss'].append(float(val_loss))
            history['val_acc'].append(float(val_acc))
        else:
            print(f'epoch: {epoch}, train_accuracy: {train_accuracy: .2f}, loss: {train_loss: .3f}')
    if save_last_weights_path:  
        torch.save(model.state_dict(), save_last_weights_path)
        print(f'Saved successfully last weights to:', save_last_weights_path)
    return model, history


class ResNetLSTM(nn.Module):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((240, 240)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    def __init__(self, num_classes=50, num_frames=20, hidden_size=128, num_lstm_layers = 2, backbone_name = 'resnet101', transform = None):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained = True, features_only = True)
        self.adap = nn.AdaptiveAvgPool2d((2,2))
        
        self.lstm = nn.LSTM(2048, hidden_size, num_lstm_layers, batch_first = True)
        
        self.fc = nn.Linear(hidden_size, num_classes)
        self.name = "ResNet LSTM"
        self.accuracy = 0.84
        self.num_frames= num_frames
        if transform is not None:
            self.transform = transform

            
    def forward(self, x):
        'x: batch, num_frames, channels, height, width'
        batch, num_frames, channels, height, width = x.shape
        
        x = torch.reshape(x, (-1, *x.shape[2:]))
        
        x1,x2,x3,x4,x5 = self.backbone(x)
        
        x = self.adap(x3)
        
        x = nn.Flatten()(x)
        
        x = torch.reshape(x, (batch, num_frames, -1))
        
        x, (h_n, c_n) = self.lstm(x)
        
        x = h_n[-1, ...]
        
        x = self.fc(x)
        
        return x

    def load_pretrained(self, dir='torch_models/pretrained/ResNetLSTM/best_weights.pt', device='cpu'):
        # self.load_state_dict(torch.load(dir), map_location=torch.device(device))
        self.load_state_dict(torch.load(dir, map_location=torch.device(device), pickle_module=pickle))
        self.eval()

    def prepare_data(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        window = max(int(frames_count/self.num_frames), 1)
        
        for i in range(self.num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * window)
            ret, frame = cap.read()
            if not ret:
                break
            fixed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(fixed_frame)
            # frames.append(frame)

        frames = torch.stack([self.transform(frame) for frame in frames])

        cap.release()
        return frames

    def predict(self, x):
        predicted_labels_probabilities = self(x[None, :, :, :, :])
        _, predicted_label = torch.max(predicted_labels_probabilities, dim = 1)
        return predicted_label
    
class ViTModel(ViT):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((240, 240)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    def __init__(self, image_size = 240,          # image size
        num_frames = 20,               # number of frames
        image_patch_size = 16,     # image patch size
        frame_patch_size = 2,      # frame patch size
        num_classes = 50,
        dim = 1024,
        depth = 6,
        heads = 8,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1):
        
        super().__init__(image_size = 240,          # image size
        frames = num_frames,               # number of frames
        image_patch_size = image_patch_size,     # image patch size
        frame_patch_size = frame_patch_size,      # frame patch size
        num_classes = num_classes,
        dim = dim,
        depth = depth,
        heads = heads,
        mlp_dim = mlp_dim,
        dropout = dropout,
        emb_dropout = emb_dropout)

        self.num_frames = num_frames
        self.name = "ViT Model"
        self.accuracy = 0.74

    def load_pretrained(self, dir='torch_models/pretrained/Vit/best_weights.pt', device='cpu'):
        self.load_state_dict(torch.load(dir, map_location=torch.device(device), pickle_module=pickle))
        self.eval()
        
    def prepare_data(self, video_path):
            cap = cv2.VideoCapture(video_path)
            frames = []
            frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            window = max(int(frames_count/self.num_frames), 1)
            
            for i in range(self.num_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i * window)
                ret, frame = cap.read()
                if not ret:
                    break
                fixed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(fixed_frame)
                # frames.append(frame)

            frames = torch.stack([self.transform(frame) for frame in frames])

            cap.release()
            return frames

    def predict(self, x):
        
        predicted_labels_probabilities = self((x[None, :, :, :, :]).reshape([-1, 3, 20, 240, 240]))
        _, predicted_label = torch.max(predicted_labels_probabilities, dim = 1)
        return predicted_label
    
class ConvLSTM:
    def __init__(self,num_classes=50, SEQUENCE_LENGTH: int = 20, IMAGE_HEIGHT: int =64, IMAGE_WIDTH: int = 64):
        self.accuracy = 0.58
        self.name = "Conv LSTM"
        self.SEQUENCE_LENGTH: int = SEQUENCE_LENGTH
        self.IMAGE_HEIGHT = IMAGE_HEIGHT
        self.IMAGE_WIDTH = IMAGE_WIDTH
        self.num_classes = num_classes
        self.model = self.create_convlstm_model()

    def predict(self, x):
        x = np.asarray(x)
        return np.argmax(self.model.predict(np.expand_dims(x, axis = 0))[0])
    
    def create_convlstm_model(self):
        '''
        This function will construct the required convlstm model.
        Returns:
            model: It is the required constructed convlstm model.
        '''

        # We will use a Sequential model for model construction
        model = Sequential()

        # Define the Model Architecture.
        ########################################################################################################################
        IMAGE_HEIGHT = self.IMAGE_HEIGHT 
        IMAGE_WIDTH =  self.IMAGE_WIDTH
        SEQUENCE_LENGTH = self.SEQUENCE_LENGTH
        model.add(ConvLSTM2D(filters = 4, kernel_size = (3, 3), activation = 'tanh',data_format = "channels_last",
                            recurrent_dropout=0.2, return_sequences=True, input_shape = (SEQUENCE_LENGTH,
                                                                                        IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
        
        model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
        model.add(TimeDistributed(Dropout(0.2)))
        
        model.add(ConvLSTM2D(filters = 8, kernel_size = (3, 3), activation = 'tanh', data_format = "channels_last",
                            recurrent_dropout=0.2, return_sequences=True))
        
        model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
        model.add(TimeDistributed(Dropout(0.2)))
        
        model.add(ConvLSTM2D(filters = 14, kernel_size = (3, 3), activation = 'tanh', data_format = "channels_last",
                            recurrent_dropout=0.2, return_sequences=True))
        
        model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
        model.add(TimeDistributed(Dropout(0.2)))
        
        model.add(ConvLSTM2D(filters = 16, kernel_size = (3, 3), activation = 'tanh', data_format = "channels_last",
                            recurrent_dropout=0.2, return_sequences=True))
        
        model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
        #model.add(TimeDistributed(Dropout(0.2)))
        
        model.add(Flatten()) 
        
        model.add(Dense(self.num_classes, activation = "softmax"))
        
        ########################################################################################################################
        
        # Display the models summary.
        # model.summary()
        
        # Return the constructed convlstm model.
        return model

    def load_pretrained(self, dir='torch_models/pretrained/ConvLSTM/convlstm_model___Date_Time_2023_11_19__11_03_26___Loss_2.0311477184295654___Accuracy_0.5669999718666077.h5', device='cpu'):
        self.model.load_weights(dir)

    def prepare_data(self, video_path):
        SEQUENCE_LENGTH = self.SEQUENCE_LENGTH
        IMAGE_HEIGHT = self.IMAGE_HEIGHT 
        IMAGE_WIDTH =  self.IMAGE_WIDTH
        video_reader = cv2.VideoCapture(video_path)
        
        # Declare a list to store video frames.
        frames_list = []
        # Get the total number of frames in the video.
        video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate the the interval after which frames will be added to the list.
        skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)
        for frame_counter in range(SEQUENCE_LENGTH):
            # Set the current frame position of the video.
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

            # Reading the frame from the video. 
            success, frame = video_reader.read() 

            # Check if Video frame is not successfully read then break the loop
            if not success:
                break

            # Resize the Frame to fixed height and width.
            resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
            normalized_frame = resized_frame / 255
        
            # Append the normalized frame into the frames list
            frames_list.append(normalized_frame)
        # Release the VideoCapture object. 
        video_reader.release()

        return frames_list

    def __call__(self, x):
        x = np.asarray(x)
        self.model.predict(np.expand_dims(x, axis = 0))[0]
        
class LRCN:
    def __init__(self, num_classes=50, SEQUENCE_LENGTH: int = 20, IMAGE_HEIGHT: int =64, IMAGE_WIDTH: int = 64):
        self.name = "LRCN"
        self.accuracy = 0.67

        self.num_classes=num_classes
        self.SEQUENCE_LENGTH = SEQUENCE_LENGTH
        self.IMAGE_HEIGHT = IMAGE_HEIGHT
        self.IMAGE_WIDTH = IMAGE_WIDTH

        
        self.model = self.create_LRCN_model()
        
    def __call__(self, x):
        x = np.asarray(x)
        self.model.predict(np.expand_dims(x, axis = 0))[0]

    def create_LRCN_model(self):
        '''
        This function will construct the required LRCN model.
        Returns:
            model: It is the required constructed LRCN model.
        '''

        # We will use a Sequential model for model construction.
        model = Sequential()
        
        # Define the Model Architecture.
        ########################################################################################################################
        
        model.add(TimeDistributed(Conv2D(16, (3, 3), padding='same',activation = 'relu'),
                                input_shape = (self.SEQUENCE_LENGTH, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 3)))
        
        model.add(TimeDistributed(MaxPooling2D((4, 4)))) 
        model.add(TimeDistributed(Dropout(0.25)))
        
        model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same',activation = 'relu')))
        model.add(TimeDistributed(MaxPooling2D((4, 4))))
        model.add(TimeDistributed(Dropout(0.25)))
        
        model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2))))
        model.add(TimeDistributed(Dropout(0.25)))
        
        model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2))))
        #model.add(TimeDistributed(Dropout(0.25)))
                                        
        model.add(TimeDistributed(Flatten()))
                                        
        model.add(LSTM(32))
                                        
        model.add(Dense(self.num_classes, activation = 'softmax'))

        ########################################################################################################################

        # Display the models summary.
        # model.summary()
        
        # Return the constructed LRCN model.
        return model

    def prepare_data(self, video_path):
        SEQUENCE_LENGTH = self.SEQUENCE_LENGTH
        IMAGE_HEIGHT = self.IMAGE_HEIGHT 
        IMAGE_WIDTH =  self.IMAGE_WIDTH
        video_reader = cv2.VideoCapture(video_path)
        
        # Declare a list to store video frames.
        frames_list = []
        # Get the total number of frames in the video.
        video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate the the interval after which frames will be added to the list.
        skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)
        for frame_counter in range(SEQUENCE_LENGTH):
            # Set the current frame position of the video.
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

            # Reading the frame from the video. 
            success, frame = video_reader.read() 

            # Check if Video frame is not successfully read then break the loop
            if not success:
                break

            # Resize the Frame to fixed height and width.
            resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
            normalized_frame = resized_frame / 255
        
            # Append the normalized frame into the frames list
            frames_list.append(normalized_frame)
        # Release the VideoCapture object. 
        video_reader.release()

        return frames_list
            
    def load_pretrained(self, dir='torch_models/pretrained/LRCN/LRCN_model___Date_Time_2023_11_19__11_27_38___Loss_1.4021259546279907___Accuracy_0.6690000295639038.h5', device='cpu'):
        self.model.load_weights(dir)

    def predict(self, x):
        x = np.asarray(x)
        return np.argmax(self.model.predict(np.expand_dims(x, axis = 0))[0])
    