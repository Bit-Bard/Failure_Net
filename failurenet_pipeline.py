import torch
import torch.nn.functional as F
from models_def import BaseCNN_MC, FailureHead


device = "cpu"


classes = [
'airplane','automobile','bird','cat','deer',
'dog','frog','horse','ship','truck'
]


cnn_model = BaseCNN_MC()
failure_head = FailureHead()

cnn_model.load_state_dict(torch.load("C:\\Users\\dhruv\\OneDrive\\Documents\\Desktop\\Deep Learning Project\\FailureNet_2\model\\best_model.pth",map_location=device))
failure_head.load_state_dict(torch.load("C:\\Users\\dhruv\\OneDrive\\Documents\\Desktop\\Deep Learning Project\\FailureNet_2\\model\\failure_head.pth",map_location=device))

cnn_model.eval()
failure_head.eval()


def enable_mc_dropout(model):

    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()



def predict(image_tensor):

    mc_passes = 20
    mc_probs = []

    enable_mc_dropout(cnn_model)

    with torch.no_grad():

        for _ in range(mc_passes):

            logits = cnn_model(image_tensor)
            probs = F.softmax(logits,dim=1)

            mc_probs.append(probs)

    mc_probs = torch.stack(mc_probs)

    mean_probs = mc_probs.mean(dim=0)

    prediction = mean_probs.argmax(dim=1).item()
    confidence = mean_probs.max().item()

    entropy = -(mean_probs * torch.log(mean_probs+1e-8)).sum().item()

    variance = mc_probs.var(dim=0).mean().item()

    top2 = torch.topk(mean_probs,2)
    margin = (top2.values[0][0]-top2.values[0][1]).item()

    features = torch.tensor([[confidence,entropy,variance,margin]])

    failure_prob = torch.sigmoid(failure_head(features)).item()

    if failure_prob > 0.6 or entropy > 1.2 or confidence < 0.5:
        decision = "REJECT"
    else:
        decision = "ACCEPT"


    return {
        "class":classes[prediction],
        "confidence":confidence,
        "entropy":entropy,
        "failure_probability":failure_prob,
        "decision":decision
    }