from pathlib import Path
import sys
current_file_path = Path(__file__)
parent_directory = current_file_path.parent.parent
sys.path.append(str(parent_directory))

from smpy import *

file_path = "pma files/hel1.pma"
image_path = "User Guide - Two Colour/hel1_Avg/hel1_Avg_Frame.png"
image = io.imread(image_path, as_gray=True)

CH1_img_path = "User Guide - Two Colour/hel1_Avg/hel1_CH1_Avg_Frame.png"
CH2_img_path = "User Guide - Two Colour/hel1_Avg/hel1_CH2_Avg_Frame.png"

good_peaks_1,_ = find_good_peaks(CH1_img_path)
good_peaks_2_CH1,_ = find_good_peaks(CH2_img_path)

CH1_arr = np.array([[18,92], [16,213], [108,43], [106,176], [210,51], [234, 219], [366,12], [322,192], [478,106], [502,160]])
CH2_arr = np.array([[22,349], [19,470], [111,300], [108,433], [212,307], [234, 475], [367,268], [321,448], [476,361], [499,414]])

params_x, params_y = find_polyfit_params(CH1_arr, CH2_arr, degree=3)
mapped_peaks= apply_polyfit_params(good_peaks_1, params_x, params_y).astype(np.uint16)
poly_pair_count, poly_pair_CH1, poly_pair_CH2 = find_pairs(good_peaks_1, mapped_peaks, tolerance=4, Channel_count=2, shift=[-1,-10])


y_centres = np.concatenate(( poly_pair_CH1[:,0], poly_pair_CH2[:,0]))
x_centres = np.concatenate(( poly_pair_CH1[:,1], poly_pair_CH2[:,1]))
circle_array_new = draw_circle(4, y_centres, x_centres, image.shape[0])


mask = (circle_array_new == [255, 255, 0]).all(axis=-1)
if image.ndim == 2:
    image_copy = image.copy()
    image_copy = np.repeat(image[..., np.newaxis], 3, -1)
elif image.ndim==3 and image.shape[2]==3:
    image_copy = image.copy()

image_copy[mask] = [255, 255, 0]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
ax.set_position([0.01, 0.3, 0.4, 0.6]) 
ax.imshow(image_copy)
plt.axvline(x= 256, color='w', linestyle='-')
ax.grid()

scat1 = ax.scatter(poly_pair_CH1[:,1], poly_pair_CH1[:,0], s=50, facecolors='none', edgecolors='g', alpha=0)
scat2 = ax.scatter(poly_pair_CH2[:,1], poly_pair_CH2[:,0], s=50, facecolors='none', edgecolors='b', alpha=0)
ax.set_title(f"Mapped Peaks ({poly_pair_count} Pairs): Click For Plots")

scatter_data = [(scat1, poly_pair_CH1 , "CH1"), (scat2, poly_pair_CH2, "CH2")]

annot = init_annot(ax=ax)

fig.canvas.mpl_connect("button_press_event", lambda event: interactive_2CH_plots(event, file_path, fig, ax, scatter_data, y_centres, x_centres, image_copy, image, mask = (circle_array_new == [255, 255, 0]).all(axis=-1), tpf=1/5, time_interval=10, background_treatment = "DG", CH_consideration=True))
fig.canvas.mpl_connect("motion_notify_event", lambda event: interactive_2CH_plots(event, file_path, fig, ax, scatter_data, y_centres, x_centres, image_copy, image, mask = (circle_array_new == [255, 255, 0]).all(axis=-1), tpf=1/5, time_interval=10, background_treatment = "DG", CH_consideration=True))

plt.show()