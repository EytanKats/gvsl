import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def show_images(img1, img2, title, epoch, idx, output_dir, mlops_task):
    img1_np = img1.cpu().numpy()[0, 0, ...].copy()
    img2_np = img2.cpu().numpy()[0, 0, ...].copy()
    for image_slice in range(63, 64):
        f, axarr = plt.subplots(1, 2)
        plt_1 = axarr[0].imshow(img1_np[:, :, image_slice], cmap='gray')
        # plt.colorbar(plt_1, ax=axarr[0])
        plt_2 = axarr[1].imshow(img2_np[:, :, image_slice], cmap='gray')
        # plt.colorbar(plt_2, ax=axarr[1])
        plt.title(title)

        output_path = os.path.join(output_dir, f'{title}_idx{idx}_ep{epoch}.png')
        plt.savefig(output_path)
        # plt.show()
        plt.close()

        mlops_task.report_matplotlib_figure(f, f'{title}_idx{idx}', epoch)


def show_tsne(features, label, title, epoch, idx, output_dir, mlops_task):
    label1_np = label.cpu().numpy()[0]
    spleen_loc = np.random.choice(np.sum(label1_np[1, ...] == 1), 1000)
    kidney_loc = np.random.choice(np.sum(label1_np[2, ...] == 1), 1000)
    liver_loc = np.random.choice(np.sum(label1_np[6, ...] == 1), 1000)
    stomach_loc = np.random.choice(np.sum(label1_np[7, ...] == 1), 1000)

    f1_np = features.detach().cpu().numpy()[0]
    spleen_f = np.transpose(f1_np[:, label1_np[1, ...] == 1])
    kidney_f = np.transpose(f1_np[:, label1_np[2, ...] == 1])
    liver_f = np.transpose(f1_np[:, label1_np[6, ...] == 1])
    stomach_f = np.transpose(f1_np[:, label1_np[7, ...] == 1])

    spleen_f = spleen_f[spleen_loc, :]
    kidney_f = kidney_f[kidney_loc, :]
    liver_f = liver_f[liver_loc, :]
    stomach_f = stomach_f[stomach_loc, :]

    features = np.concatenate([spleen_f, kidney_f, liver_f, stomach_f])
    labels = np.concatenate([['spleen'] * 1000, ['kidney'] * 1000, ['liver'] * 1000, ['stomach'] * 1000])

    plot_name = f'{title}_idx{idx}_ep{epoch}'
    output_path = os.path.join(output_dir, f'{plot_name}.png')
    plot_tsne(features, labels, num_classes=4, title=plot_name, show=False, output_path=output_path, mlops_task=None, mlops_iteration=epoch)


def plot_tsne(features, labels, num_classes, title='T-SNE projection', show=False, output_path=None, mlops_task=None, mlops_iteration=0):

    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(features)

    df = pd.DataFrame()
    df["y"] = labels
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]

    fig, ax = plt.subplots(figsize=(16, 10))
    sns.scatterplot(ax=ax, x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", num_classes),
                    data=df).set(title=title)

    if output_path:
        plt.savefig(output_path)

    if show:
        plt.show()

    if mlops_task:
        mlops_task.report_matplotlib_figure(fig, title, mlops_iteration)

    plt.close()