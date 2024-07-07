## Model Architecture
The `MovieRatingNN` is a simple feed-forward neural network with the following structure:
- **Input Layer**: Receives input features (`num_features`).
- **First Hidden Layer**: Linear layer with 128 neurons, followed by ReLU activation.
- **Second Hidden Layer**: Linear layer with 64 neurons, followed by ReLU activation.
- **Output Layer**: Linear layer with one output neuron. The output passes through a sigmoid function scaled to provide values from 1 to 5, reflecting the range of possible movie ratings.

## Data Preprocessing
Inputs are normalized or encoded as required. Missing data is appropriately filled or removed before training.

## Model Artifacts
The trained model's state dictionary is saved in the following location:
**../../artifacts/simple_nn.pth**

### Loading the Model
You can load the model with the following Python code:
```python
import torch
from model import MovieRatingNN

# Initialize the model
model = MovieRatingNN(num_features=112)
model.load_state_dict(torch.load('../../artifacts/simple_nn.pth'))
model.eval()

# Making a Prediction
dummy_input = torch.randn(1, 112)  # Example tensor for one sample
prediction = model(dummy_input)
print("Predicted Rating:", prediction.item())