# Setup 


## start up back-end

```bash
cd backend/app
pip install -r requiremetns.txt
```

start up command 

```bash
python main.py
```

## start up front-end

```bash
cd frontend
npm start
```

## Start up with docker

If you have Docker installed on your computer, you can run the following docker commands 

```Bash
cd backend
docker build -t backend . 
docker run -d -p 8000:8000 backend
```

```Bash
cd frontend 
docker build -t frontend .
docker run -p 3000:3000 frontend
```

[Access SwaggerUI](http://localhost:8000/api/docs)

# User Interface

The startup screen shows up after activating the front end. 
   
![](docs_images/1.png)

## Cell Extraction

This section allows you to extract cells from arbitrary ND2 files that consist of up to three layers (e.g., PH, GFP, and YFP).

1. The first thing you will see in this section is the ND2 table shown below.
   ![](docs_images/2.png)
   First, you need to upload an ND2 file from your computer using the `SELECT ND2 FILE` button.
   ![](docs_images/3.png)

2. After selecting an ND2 file to upload, press the `UPLOAD` button to actually submit the file to the backend. This process takes approximately 30 seconds.
   ![](docs_images/4.png)
   If the file is successfully submitted, a pop-up alert will appear like this, and the filename will be added to the list as well.
   ![](docs_images/5.png)

3. The next step is to press the `EXTRACT CELLS` button.
   ![](docs_images/6.png)

4. After entering the cell extraction section, you will see the parameter input fields.
   ![](docs_images/7.png)
   Press the `EXTRACT CELLS` button when you have finished inputting all the parameters, and the cell extraction process will begin on the backend. This will take approximately 1 minute.

   <!-- Add parameter descriptions here. -->
5. When all processes are finished, detected cell contours will appear on the PH image with frames extracted from the input ND2 file. This is where you can check if the contour detection was successful. If not, you can adjust the parameters and press the `RE-EXTRACT CELLS` button. If everything looks fine, press the `GO TO DATABASE` button to proceed to the next step, which is labeling each cell.
   ![](docs_images/8.png)

6. After moving on to the section below the cell extraction section, you can access the database with the same prefix as the input ND2 file. The automatically generated database contains all the cell information and is renamed with the postfix `-uploaded` as a tag. The `Mark as Complete` button is disabled at this point because all the extracted cells are labeled as `N/A`.
   ![](docs_images/9.png)

7. When you press the `ACCESS DATABASE` button, you will see the cell labeling section as shown below.
   ![](docs_images/10.png)
   In this section, you can label all the cells extracted from the previous section with labels ["N/A", "1", "2", "3"]. You can also input these labels from the keyboard using ["n", "1", "2", "3"], and "Enter" corresponds to the "Next" UI button. Note that labels for each cell are automatically updated on the backend as soon as you select one in the list.

8. After labeling all the cells with arbitrary labels, you can go back to the `Database console` to find that the `Mark as Complete` button is enabled. 
   ![](docs_images/12.png)
   When you mark the database as complete, the database name is renamed to a new one with `-completed` as a postfix, and the labeled databases are downloadable(i.e., press the `EXPORT DATABASE` button to download the file.)
   ![](docs_images/db.png)
   

# Data Analyses

After obtaining the cell database with selected labels for each cell, in other words, after pressing `mark as completed`, it will show up in the `completed` section of the console. (Note that the `Validated` tag is only for admin usage (i.e., uncontrollable from the frontend))
   ![](docs_images/13.png)

When you press the `Access database` button, you will go to a similar section as the cell labeling console. (It is actually the same but without the labeling function.)
   ![](docs_images/14.png)

The next action you will take is to select the cells with a specific label in the list. 
   ![](docs_images/15.png)

   In this example, we will be using cells labeled with `1` as shown below.
   ![](docs_images/16.png)

## Parameters 

If you uncheck both the `Contour` and `Scale bar` checkboxes, you will see the raw images of the cells. 
   ![](docs_images/17.png)

If you check both the `Contour` and `Scale bar` checkboxes, you will see the cell images with their contours and a scale bar. 
   ![](docs_images/18.png)

If you increase or decrease the `Brightness Factor`, you can adjust the brightness of each pixel to see the localization of the fluorescence, etc. 
   ![](docs_images/19.png)

The `Manual Label` section just shows you the label of the cell and never updates it in this section because the labeling process is already done beforehand.

## Graph Section

There is a graph section in the middle column, which is set to `Light` as the default value. It shows the contour of the cell with its center at the center of the figure.
   ![](docs_images/20.png)

The draw modes consist of these three modes:

![](docs_images/21.png)

If you choose `Replot`, it shows the replotted figure of the cell. 

![](docs_images/22.png)

If you choose `Peak-path`, it shows the peak-path figure of the cell. 

![](docs_images/23.png)

`Polyfit degree` is a meta-parameter for the polynomial fitting of the cell center curve, but it is generally good practice to leave it at four.

# Morphoengines 

There are four Morphoengines that conduct morphological analyses of the cells.

![](docs_images/24.png)

If you choose `Morphoengine 2.0`, it shows you the morphological parameters for each cell. 
(The volume calculation algorithms are a work in progress.)

![](docs_images/25.png)

If you choose `MedianEngine`, it shows you the normalized median of the pixels inside the cell, highlighted with a red dot, and the other blue dots are those cells labeled as `1` in this case.

![](docs_images/26.png)

`MeanEngine` performs a similar function but for the normalized mean of the pixels inside the cells.

![](docs_images/27.png)

`AreaFractionEngine` segments the HU-GFP signal within each cell and compares the resulting nucleoid area fraction with other cells sharing the same label, visualizing the position of the current cell within the distribution.

![](docs_images/26.png)

`HeatmapEngine` qualitatively visualizes the localization of the fluorescence at relative positions from the edge of the cell.

![](docs_images/28.png)