import torch
from torch import nn
import torch.nn.functional as F


class CharModel(nn.Module):

    def __init__(self, all_chars, num_hidden=256, num_layers=4, drop_prob=0.5, use_gpu=False):

        # SET UP ATTRIBUTES
        super().__init__()
        self.drop_prob = drop_prob
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.use_gpu = use_gpu

        # CHARACTER SET, ENCODER, and DECODER
        self.all_chars = all_chars
        self.decoder = dict(enumerate(all_chars))
        self.encoder = {char: ind for ind, char in decoder.items()}

        self.lstm = nn.LSTM(len(self.all_chars), num_hidden, num_layers, dropout=drop_prob, batch_first=True)

        self.dropout = nn.Dropout(drop_prob)

        self.fc_linear = nn.Linear(num_hidden, len(self.all_chars))

    def forward(self, x, hidden):

        lstm_output, hidden = self.lstm(x, hidden)

        drop_output = self.dropout(lstm_output)

        drop_output = drop_output.contiguous().view(-1, self.num_hidden)

        final_out = self.fc_linear(drop_output)

        return final_out, hidden

    def hidden_state(self, batch_size):
        '''
        Used as separate method to account for both GPU and CPU users.
        '''

        if self.use_gpu:

            hidden = (torch.zeros(self.num_layers, batch_size, self.num_hidden).cuda(),
                      torch.zeros(self.num_layers, batch_size, self.num_hidden).cuda())
        else:
            hidden = (torch.zeros(self.num_layers, batch_size, self.num_hidden),
                      torch.zeros(self.num_layers, batch_size, self.num_hidden))

        return hidden

model = CharModel(
    all_chars=all_characters,
    num_hidden=512,
    num_layers=3,
    drop_prob=0.5,
    use_gpu=True,
)
model.load_state_dict(torch.load(model_name))
model.eval()


def predict_next_char(model, char, hidden=None, k=1):
    # Encode raw letters with model
    encoded_text = model.encoder[char]

    # set as numpy array for one hot encoding
    # NOTE THE [[ ]] dimensions!!
    encoded_text = np.array([[encoded_text]])

    # One hot encoding
    encoded_text = one_hot_encoder(encoded_text, len(model.all_chars))

    # Convert to Tensor
    inputs = torch.from_numpy(encoded_text)

    # Check for CPU
    if (model.use_gpu):
        inputs = inputs.cuda()

    # Grab hidden states
    hidden = tuple([state.data for state in hidden])

    # Run model and get predicted output
    lstm_out, hidden = model(inputs, hidden)

    # Convert lstm_out to probabilities
    probs = F.softmax(lstm_out, dim=1).data

    if (model.use_gpu):
        # move back to CPU to use with numpy
        probs = probs.cpu()

    # k determines how many characters to consider
    # for our probability choice.
    # https://pytorch.org/docs/stable/torch.html#torch.topk

    # Return k largest probabilities in tensor
    probs, index_positions = probs.topk(k)

    index_positions = index_positions.numpy().squeeze()

    # Create array of probabilities
    probs = probs.numpy().flatten()

    # Convert to probabilities per index
    probs = probs / probs.sum()

    # randomly choose a character based on probabilities
    char = np.random.choice(index_positions, p=probs)

    # return the encoded value of the predicted char and the hidden state
    return model.decoder[char], hidden


def generate_text(model, size, seed='The', k=1):
    # CHECK FOR GPU
    if (model.use_gpu):
        model.cuda()
    else:
        model.cpu()

    # Evaluation mode
    model.eval()

    # begin output from initial seed
    output_chars = [c for c in seed]

    # intiate hidden state
    hidden = model.hidden_state(1)

    # predict the next character for every character in seed
    for char in seed:
        char, hidden = predict_next_char(model, char, hidden, k=k)

    # add initial characters to output
    output_chars.append(char)

    # Now generate for size requested
    for i in range(size):
        # predict based off very last letter in output_chars
        char, hidden = predict_next_char(model, output_chars[-1], hidden, k=k)

        # add predicted character
        output_chars.append(char)

    # return string of predicted text
    return ''.join(output_chars)

print(generate_text(model, 1000, seed='The ', k=3))