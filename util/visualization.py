from datetime import datetime

import matplotlib.pyplot as plt

from util.util import setup_logger

current_datetime = datetime.now()
vis_logger = setup_logger('visualization',
                          "log/visual_{}_{}_{}_{}.log".format(current_datetime.year, current_datetime.month,
                                                              current_datetime.day,
                                                              current_datetime.hour))


def plot_inverted(dataset, dataset_unscaled, dataset_inverted, DESIRED_OUTPUT, OUTPUT_TOLERANCE):
    vis_logger.info("Started plot_inverted method")
    dataset_original = dataset_unscaled.copy().values.tolist();
    dataset_original_df = dataset_unscaled.copy()
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1, figsize=(20, 6))
    x_number_list_o = [values[0] for values in dataset_original]
    # y axis value list.
    y_number_list_o = [values[1] for values in dataset_original]
    # Draw point based on above x, y axis values.
    ax1.scatter(x_number_list_o, y_number_list_o)
    ax1.set_xlim([0 - 5, dataset["Position X"].max() + 5])
    ax1.set_ylim([0 - 5, dataset["Position Y"].max() + 5])
    # Set chart title.
    ax1.title.set_text("Original coordinates of the dataset")
    # Set x, y label text.
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    x_number_list = [values[0] for values in dataset_original if
                     values[8] > DESIRED_OUTPUT - OUTPUT_TOLERANCE and values[8] < DESIRED_OUTPUT + OUTPUT_TOLERANCE]
    # y axis value list.
    y_number_list = [values[1] for values in dataset_original if
                     values[8] > DESIRED_OUTPUT - OUTPUT_TOLERANCE and values[8] < DESIRED_OUTPUT + OUTPUT_TOLERANCE]
    # Draw point based on above x, y axis values.
    ax2.scatter(x_number_list, y_number_list)
    ax2.set_xlim([0 - 5, dataset["Position X"].max() + 5])
    ax2.set_ylim([0 - 5, dataset["Position Y"].max() + 5])
    # Set chart title.
    ax2.title.set_text("Original coordinates reduced by currently detected WiFi RSSI")
    # Set x, y label text.
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    x_number_list = [values[0] for values in dataset_inverted]
    # y axis value list.
    y_number_list = [values[1] for values in dataset_inverted]
    ax3.scatter(x_number_list, y_number_list, color="r")
    ax3.set_xlim([0 - 5, dataset["Position X"].max() + 5])
    ax3.set_ylim([0 - 5, dataset["Position Y"].max() + 5])
    # Set chart title.
    ax3.title.set_text("Inverted coordinates by genetic algorithm")
    # Set x, y label text.
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    plt.savefig('coordinatesinverted.pdf')
    plt.show()
    vis_logger.info("Done plot_inverted method")
