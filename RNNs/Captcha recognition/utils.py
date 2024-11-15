import torch


def remove_duplicates(x):
    if len(x) < 2:
        return x

    fin = ""
    for j in x:
        if fin == "":
            fin = j
        else:
            if j == fin[-1]:
                continue
            else:
                fin = fin + j

    return fin


def decode_predictions(preds, encoder):
    preds = preds.permute(1, 0, 2)  # from t_step, bs, class -> bs, t_step, class
    print(preds.size())
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()
    cap_preds = []

    for bs in range(preds.shape[0]):
        temp = []
        for value in preds[bs, :]:
            k = value - 1
            if k == -1:
                temp.append("~")
            else:
                p = encoder.inverse_transform([k])[0]
                temp.append(p)

        tp = "".join(temp).replace("~", "")
        # cap_preds.append(tp)
        cap_preds.append((remove_duplicates(tp)))

    return cap_preds
