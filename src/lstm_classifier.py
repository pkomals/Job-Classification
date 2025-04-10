import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

class JobDataset(Dataset):
    def __init__(self, texts, labels, vectorizer, max_length=200):
        self.texts = texts
        self.labels = labels
        self.vectorizer = vectorizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Ensure text is a string
        if isinstance(text, np.ndarray):
            text = str(text[0]) if text.size > 0 else ""
        elif not isinstance(text, str):
            text = str(text)
        
        # Convert text to sequence using the vectorizer
        sequence = self.vectorizer.transform([text]).toarray()[0]
        
        # Pad sequence if needed
        if len(sequence) < self.max_length:
            padded = np.zeros(self.max_length, dtype=np.float32)
            padded[:len(sequence)] = sequence
            sequence = padded
        else:
            sequence = sequence[:self.max_length]
        
        return torch.FloatTensor(sequence), torch.LongTensor([label])

class LSTMClassifier:
    def __init__(self, max_words=10000, max_sequence_length=200, embedding_dim=100):
        self.max_words = max_words
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.vectorizer = None
        self.label_encoder = LabelEncoder()
        self.model = None
        self.classes_ = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _create_model(self, num_classes):
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_dim, num_classes):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=2, 
                                  batch_first=True, dropout=0.3, bidirectional=True)
                self.fc = nn.Sequential(
                    nn.Linear(hidden_dim * 2, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, num_classes)
                )
                
            def forward(self, x):
                # Reshape input for LSTM (batch_size, seq_len, input_size)
                x = x.unsqueeze(1)  # Add sequence length dimension
                lstm_out, _ = self.lstm(x)
                last_hidden = lstm_out[:, -1, :]
                output = self.fc(last_hidden)
                return output

        model = LSTMModel(
            input_size=self.max_sequence_length,
            hidden_dim=128,
            num_classes=num_classes
        )
        return model.to(self.device)

    def train(self, X, y, batch_size=32, epochs=10):
        # Create vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_words,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.8,
            stop_words='english'
        )
        
        # Fit vectorizer
        X_tfidf = self.vectorizer.fit_transform(X)
        self.max_sequence_length = min(X_tfidf.shape[1], self.max_sequence_length)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        
        # Create dataset and dataloader
        dataset = JobDataset(X, y_encoded, self.vectorizer, self.max_sequence_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Create model
        self.model = self._create_model(len(self.classes_))
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y.squeeze())
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}')

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            # Convert texts to sequences
            X_tfidf = self.vectorizer.transform(X).toarray()
            X_tensor = torch.FloatTensor(X_tfidf).to(self.device)
            
            # Make predictions
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            predictions = np.argmax(probabilities, axis=1)
            
            return self.label_encoder.inverse_transform(predictions), probabilities

    def save_model(self, model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model state
        torch.save(self.model.state_dict(), model_path)
        
        # Save vectorizer and label encoder
        vectorizer_path = model_path.replace('.pt', '_vectorizer.joblib')
        label_encoder_path = model_path.replace('.pt', '_label_encoder.joblib')
        
        joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump(self.label_encoder, label_encoder_path)

    def load_model(self, model_path):
        # Load model state
        self.model = self._create_model(len(self.classes_))
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        
        # Load vectorizer and label encoder
        vectorizer_path = model_path.replace('.pt', '_vectorizer.joblib')
        label_encoder_path = model_path.replace('.pt', '_label_encoder.joblib')
        
        self.vectorizer = joblib.load(vectorizer_path)
        self.label_encoder = joblib.load(label_encoder_path)
        self.classes_ = self.label_encoder.classes_ 