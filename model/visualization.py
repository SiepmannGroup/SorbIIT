import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider


def combine_img(left, right, line):
    w, h = left.shape[1], left.shape[0]
    #line.set_xdata([(w+1)//2, (w+1)//2])
    #line.set_ydata([0, h])
    return np.concatenate([left[:, :(w+1)//2], right[:,(w+1)//2:]], 1)

def visualize_slice(arr, arr_split=None, **kwargs):
    i = -1
    axis = 2
    ch = 0

    fig, ax = plt.subplots(figsize=(4, 2.5))
    if arr_split is not None:
        line, = ax.plot([(arr.shape[2] + 1) // 2, (arr.shape[2] + 1) // 2], [0, arr.shape[1]], color="0.5")
        axes_image = ax.imshow(combine_img(
            arr[ch, :, :, i], arr_split[ch, :, :, i], line,
        ), **kwargs)
    else:
        axes_image = ax.imshow(arr[ch, :, :, i], **kwargs)
    plt.subplots_adjust(left=0.3, bottom=0.05)

    ax_slider = plt.axes([0.25, 0.1, 0.0225, 0.8])
    ax_radio = plt.axes([0.1, 0.05, 0.1, 0.2])
    ax_radio2 = plt.axes([0.1, 0.35, 0.1, 0.6])
    slider = Slider(
        ax=ax_slider,
        label="Value",
        valmin=-1,
        valmax=max(arr.shape[1:]) - 1,
        valinit=i,
        orientation="vertical",
        valstep=1,
        valfmt="%d",
    )
    radio_dim = RadioButtons(
        ax_radio, ('X', 'Y', 'Z'), active=2)
    radio_channel = RadioButtons(
        ax_radio2, list(range(arr.shape[0])), active=0)

    def update():
        if i < 0:
            if arr_split is not None:
                axes_image.set_data(combine_img(
                    arr[ch].mean(i), arr_split[ch].mean(i), line
                ))
            else:
                axes_image.set_data(arr[ch].mean(i))
        if arr_split is not None:
            if axis == 0:
                axes_image.set_data(combine_img(
                    arr[ch, i, :, :], arr_split[ch, i, :, :], line
                ))
            elif axis == 1:
                axes_image.set_data(combine_img(
                    arr[ch, :, i, :], arr_split[ch, :, i, :], line
                ))
            else:
                axes_image.set_data(combine_img(
                    arr[ch, :, :, i], arr_split[ch, :, :, i], line
                ))
        else:
            if axis == 0:
                axes_image.set_data(arr[ch, i, :, :])
            elif axis == 1:
                axes_image.set_data(arr[ch, :, i, :])
            else:
                axes_image.set_data(arr[ch, :, :, i])
        fig.canvas.draw_idle()

    def update_slice(val):
        nonlocal i
        i = min(int(val), arr.shape[axis + 1] - 1)
        update()
        
    def update_axis(val):
        nonlocal axis
        axis = ord(val) - ord("X")
        update()
    
    def update_channel(val):
        nonlocal ch
        ch = int(val)
        update()

    slider.on_changed(update_slice)
    radio_dim.on_clicked(update_axis)
    radio_channel.on_clicked(update_channel)

    plt.show()

if __name__ == "__main__":
    img = np.random.randn(4, 64, 64, 64)
    visualize_slice(img)