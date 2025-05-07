from pathlib import Path
import sys
current_file_path = Path(__file__)
parent_directory = current_file_path.parent.parent
sys.path.append(str(parent_directory))

from smpy import *

file_path = "pma files/hel1.pma"
image_path = "User Guide - Two Colour/hel1_Avg/hel1_Avg_Frame.png"
image = io.imread(image_path, as_gray=True)

#Chanel image paths
CH1_img_path = "User Guide - Two Colour/hel1_Avg/hel1_CH1_Avg_Frame.png"
CH2_img_path = "User Guide - Two Colour/hel1_Avg/hel1_CH2_Avg_Frame.png"

good_peaks_1,_ = find_good_peaks(CH1_img_path)
good_peaks_2_CH1,_ = find_good_peaks(CH2_img_path)


good_peaks_1_CH2 = shift_peaks(good_peaks_1)
good_peaks_2_CH2 = shift_peaks(good_peaks_2_CH1)
good_peaks_2_shift = shift_peaks(good_peaks_2_CH2, shift=[-1, -10])

fig = plt.figure(figsize=(8, 8))
ax = fig.subplots()
plt.axhline(y= 102, color='w', linestyle='--')  
plt.axhline(y= 204, color='w', linestyle='--')
plt.axhline(y= 308, color='w', linestyle='--')
plt.axhline(y= 410, color='w', linestyle='--')

plt.axvline(x= 128, color='w', linestyle='--')
plt.axvline(x= 384, color='w', linestyle='--')

plt.axvline(x= 256, color='w', linestyle='-')
plt.suptitle("CH1 and CH2 Identified Peaks", fontsize=16)
plt.title("Hover over points to see peak index and coordinates \n Click on peaks to print peak info in terminal \n Indentify and select corresponding peaks in CH1 and CH2 from each section of the image", fontsize=10)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.imshow(image, cmap="gray")

scat1 = ax.scatter(good_peaks_1[:, 1], good_peaks_1[:, 0], s=50, facecolors='none', edgecolors='orange', label='Peaks from CH1')
scat2 = ax.scatter(good_peaks_2_shift[:, 1], good_peaks_2_shift[:, 0], s=50, facecolors='none', edgecolors='b', label='Peaks from CH2')
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.15))

annot = init_annot(ax=ax)

scatter_data = [(scat1, good_peaks_1, "CH1"), (scat2, good_peaks_2_shift, "CH2")]

fig.canvas.mpl_connect("motion_notify_event", lambda event: print_coords_trigger(event, fig, scatter_data))
fig.canvas.mpl_connect("button_press_event", lambda event: print_coords_trigger(event, fig, scatter_data))

plt.show()



