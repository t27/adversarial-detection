import numpy as np
import torch
import torchvision
from torchvision.transforms import ToPILImage
from torchvision import transforms
import pandas as pd
from io import BytesIO
import base64
import infer
import adversarial_detector
from tqdm import tqdm

to_pil = ToPILImage()
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.5, 0.5, 0.5])
unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

real_train_loader, real_dev_loader = infer.get_cifar_loader()

adv_train_loader, adv_dev_loader = infer.adv_loader()


def image_base64(im):
    with BytesIO() as buffer:
        im.save(buffer, "png")
        return base64.b64encode(buffer.getvalue()).decode()


def run():
    # dataloader = adv_dev_loader
    dataloader = real_dev_loader
    basemodel = infer.get_base_model()
    basemodel.eval()
    if torch.cuda.is_available():
        basemodel = basemodel.cuda()
    adv_models = []
    for i in range(4):
        model = adversarial_detector.get_trained_model(i + 1)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        adv_models.append(model)
    results_names = ["image", "actual", "prediction", "c1", "c2", "c3", "c4"]
    results = []
    for idx, (image, target) in tqdm(enumerate(dataloader)):
        if torch.cuda.is_available():
            image = image.cuda()
        output, c1, c2, c3, c4 = basemodel(image, layer_outputs=True)
        prediction = torch.argmax(output, dim=1).detach().cpu().numpy()
        batch_size = image.shape[0]
        adv_model_inps = [c1, c2, c3, c4]
        adv_results = []
        for i, model in enumerate(adv_models):
            res = model(adv_model_inps[i]).squeeze()
            is_adv = torch.argmax(res, dim=1).bool()
            adv_results.append(is_adv.detach().cpu().numpy())
        image = image.detach().cpu()
        target = target.cpu().numpy()
        for i in range(batch_size):
            results.append(
                [
                    image_base64(to_pil(unnormalize(image[i]))),
                    target[i],
                    prediction[i],
                    adv_results[0][i],
                    adv_results[1][i],
                    adv_results[2][i],
                    adv_results[3][i],
                ]
            )
    print("Generating and Saving Dataframe")
    df = pd.DataFrame(results, columns=results_names)
    df.to_csv("dev_real_dataset_results.csv", index=False)


if __name__ == "__main__":
    run()
