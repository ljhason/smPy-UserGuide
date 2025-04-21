from pathlib import Path
import sys
current_file_path = Path(__file__)
parent_directory = current_file_path.parent.parent
sys.path.append(str(parent_directory))

from smpy import *

file_path = "pma files/Synthetic_hel1.pma"
image_path = "User Guide - Three Colour/Synthetic_hel1_Avg/Synthetic_hel1_Avg_Frame.png"
image = io.imread(image_path, as_gray=True)

CH1_img_path = "User Guide - Three Colour/Synthetic_hel1_Avg/Synthetic_hel1_CH1_Avg_Frame.png"
CH2_img_path = "User Guide - Three Colour/Synthetic_hel1_Avg/Synthetic_hel1_CH2_Avg_Frame.png"
CH3_img_path = "User Guide - Three Colour/Synthetic_hel1_A3CH_seriesvg/Synthetic_hel1_CH3_Avg_Frame.png"

good_peaks_1_CH1, _ = find_good_peaks(CH1_img_path)
good_peaks_2_CH1, _ = find_good_peaks(CH2_img_path)
good_peaks_3_CH1, _ = find_good_peaks(CH3_img_path)

shift_CH2 = [-1,-10]
shift_CH3 = [-1,-11]
good_peaks_2_CH1_shift = shift_peaks(good_peaks_2_CH1, shift=shift_CH2)
good_peaks_3_CH1_shift = shift_peaks(good_peaks_3_CH1, shift=shift_CH3)
good_peaks_2_CH2 = shift_peaks(good_peaks_2_CH1_shift, [0, 171])
good_peaks_3_CH3 = shift_peaks(good_peaks_3_CH1_shift, [0, 342])

CH1_array = np.array([[63, 15], [51, 115], [120, 50], [108, 125], [210, 51], [228, 134], [322, 42], [327, 140], [422, 45], [420, 143]])
CH2_array = np.array([[67, 186], [54, 288], [123,222], [110, 297], [212, 222], [229, 306], [323, 213], [327,311], [422, 216], [419, 313]])
CH3_array = np.array([[67, 356], [54, 458], [123, 392], [110,467], [212, 392], [229, 476], [323, 383], [327,481], [422, 386], [419, 483]])

params_x_CH12, params_y_CH12 = find_polyfit_params(CH1_array, CH2_array, degree=3)
params_x_CH23, params_y_CH23 = find_polyfit_params(CH1_array, CH3_array, degree=3)

mapped_CH2 = apply_polyfit_params(good_peaks_1_CH1, params_x_CH12, params_y_CH12).astype(np.uint16)
mapped_CH3 = apply_polyfit_params(good_peaks_1_CH1, params_x_CH23, params_y_CH23).astype(np.uint16)

count_trip, trip_CH1, trip_CH2, trip_CH3 = find_trip(good_peaks_1_CH1, mapped_CH2, mapped_CH3, tolerance=4, shift_CH2=shift_CH2, shift_CH3=shift_CH3)

y_centres = np.concatenate((trip_CH1[:,0], trip_CH2[:,0], trip_CH3[:,0]))
x_centres = np.concatenate((trip_CH1[:,1], trip_CH2[:,1], trip_CH3[:,1]))
circle_array_new = draw_circle(4, y_centres, x_centres, image.shape[0])

mask = (circle_array_new == [255, 255, 0]).all(axis=-1)
if image.ndim == 2:
    image_copy = image.copy()
    image_copy = np.repeat(image[..., np.newaxis], 3, -1)
elif image.ndim==3 and image.shape[2]==3:
    image_copy = image.copy()

image_copy[mask] = [255, 255, 0]

fig, ax = plt.subplots(figsize=(9, 9))
ax.set_position([0.01, 0.3, 0.4, 0.6])
ax.imshow(image_copy)
plt.axvline(x= 171, color='w', linestyle='-')
plt.axvline(x= 342, color='w', linestyle='-')
ax.grid()

scat1 = ax.scatter(trip_CH1[:,1], trip_CH1[:,0], s=50, facecolors='none', edgecolors='g', alpha=0)
scat2 = ax.scatter(trip_CH2[:,1], trip_CH2[:,0], s=50, facecolors='none', edgecolors='b', alpha=0)
scat3 = ax.scatter(trip_CH3[:,1], trip_CH3[:,0], s=50, facecolors='none', edgecolors='purple', alpha=0)
ax.set_title("Mapped Peaks: Click To Display Time-Series")

scatter_data = [(scat1, trip_CH1, "CH1"), (scat2, trip_CH2, "CH2"), (scat3, trip_CH3, "CH3")]

annot = init_annot(ax=ax)

fig.canvas.mpl_connect("button_press_event", lambda event: interactive_3CH_plots(event, file_path, fig, ax, scatter_data, y_centres, x_centres, image_copy, image, mask = (circle_array_new == [255, 255, 0]).all(axis=-1), tpf=1/5, background_treatment="DG", CH_consideration="True"))
fig.canvas.mpl_connect("motion_notify_event", lambda event: interactive_3CH_plots(event, file_path, fig, ax, scatter_data, y_centres, x_centres, image_copy, image, mask = (circle_array_new == [255, 255, 0]).all(axis=-1), tpf=1/5, background_treatment="DG", CH_consideration="True"))
plt.show()
    

