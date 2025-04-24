from pathlib import Path
import sys
current_file_path = Path(__file__)
parent_directory = current_file_path.parent.parent
sys.path.append(str(parent_directory))

from smpy import *

file_path = "../pma files/Synthetic_hel1.pma"
image_path = "User Guide - Three Colour/Synthetic_hel1_Avg/Synthetic_hel1_Avg_Frame.png"
image = io.imread(image_path, as_gray=True)

CH1_img_path = "User Guide - Three Colour/Synthetic_hel1_Avg/Synthetic_hel1_CH1_Avg_Frame.png"
CH2_img_path = "User Guide - Three Colour/Synthetic_hel1_Avg/Synthetic_hel1_CH2_Avg_Frame.png"
CH3_img_path = "User Guide - Three Colour/Synthetic_hel1_Avg/Synthetic_hel1_CH3_Avg_Frame.png"

good_peaks_1_CH1, _ = find_good_peaks(CH1_img_path)
good_peaks_2_CH1, _ = find_good_peaks(CH2_img_path)
good_peaks_3_CH1, _ = find_good_peaks(CH3_img_path)

shift_CH2 = [-1,-10]
shift_CH3 = [-1,-11]
good_peaks_2_CH1_shift = shift_peaks(good_peaks_2_CH1, shift=shift_CH2)
good_peaks_3_CH1_shift = shift_peaks(good_peaks_3_CH1, shift=shift_CH3)
good_peaks_2_CH2 = shift_peaks(good_peaks_2_CH1_shift, [0, 171])
good_peaks_3_CH3 = shift_peaks(good_peaks_3_CH1_shift, [0, 342])


fig = plt.figure(figsize=(8, 8))
ax = fig.subplots()
plt.axhline(y= 102, color='w', linestyle='--')  
plt.axhline(y= 204, color='w', linestyle='--')
plt.axhline(y= 308, color='w', linestyle='--')
plt.axhline(y= 410, color='w', linestyle='--')

plt.axvline(x= 86, color='w', linestyle='--')
plt.axvline(x= 256, color='w', linestyle='--')
plt.axvline(x= 428, color='w', linestyle='--')

plt.axvline(x= 171, color='w', linestyle='-')
plt.axvline(x= 342, color='w', linestyle='-')
plt.suptitle("CH1, CH2 and CH3 Identified Peaks", fontsize=16)
plt.title("Hover over points to see peak index and coordinates \n Click on peaks to print peak info in terminal \n Indentify and select corresponding peaks in CH1, CH2, and CH3 from each section of the image", fontsize=10)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.imshow(image, cmap="gray")

scat1 = ax.scatter(good_peaks_1_CH1[:, 1], good_peaks_1_CH1[:, 0], s=50, facecolors='none', edgecolors='orange', label='Peaks from CH1')
scat2 = ax.scatter(good_peaks_2_CH2[:, 1], good_peaks_2_CH2[:, 0], s=50, facecolors='none', edgecolors='b', label='Peaks from CH2')
scat3 = ax.scatter(good_peaks_3_CH3[:, 1], good_peaks_3_CH3[:, 0], s=50, facecolors='none', edgecolors='purple', label='Peaks from CH3')
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.15))

annot = init_annot(ax=ax)

scatter_data = [(scat1, good_peaks_1_CH1, "CH1"), (scat2, good_peaks_2_CH2, "CH2"), (scat3, good_peaks_3_CH3, "CH3")]
fig.canvas.mpl_connect("motion_notify_event", lambda event: print_coords_trigger(event, fig, scatter_data))
fig.canvas.mpl_connect("button_press_event", lambda event: print_coords_trigger(event, fig, scatter_data))

plt.show()



