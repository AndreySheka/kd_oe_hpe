## Preparing

1. Download datasets 300W-LP, AFLW2000 [1]; AFLW [2]; BIWI [3] and extract its to `data/`.
Folder should be looks like:
    ```
    data/
        300W_LP.json
        AFLW.json
        AFLW2000.json
        BIWI.json
        300W_LP/
            AFW/
                AFW_134212_1_0.jpg
                AFW_134212_1_0.mat
                ...
            ...
        AFLW/
            data/
                flickr/
                    0/
                        image00002.jpg
                        image00013.jpg
                        ...
                    2/
                    3/
            aflw.sqlite
        AFLW2000/
            image00002.jpg
            image00002.mat
            ...
        BIWI/
            hpdb/
                01/
                    frame_00003_rgb.png
                    frame_00003_pose.txt
                    ...
                02/
                ...
                24/
    ```

    [1] 300W-LP and AFLW2000 datasets: http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm

    [2] AFLW dataset: https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/

    [3] BIWI Kinect Head Pose dataset: https://icu.ee.ethz.ch/research/datsets.html

2. Install python modules:
    ```
    pip3 install -r requirements.txt
    ```

## Evaluation
Download the models from the [link](https://www.dropbox.com/sh/6sq3x7j0j160db9/AACrvNZu5xdUFh33POScjonoa?dl=0) and place them in the `models/` folder. Folder should be looks like:
    ```
    models/
        ap1/
            resnet18.pth
            resnet34.pth
            resnet50.pth
            resnet101.pth
            resnet152.pth
        ap2/
            resnet18.pth
            resnet34.pth
            resnet50.pth
            resnet101.pth
            resnet152.pth
        ap3/
            resnet18.pth
            resnet34.pth
            resnet50.pth
            resnet101.pth
            resnet152.pth
    ```


For evaluation use script `src/main.py`:

```
python3 src/main.py --protocol 1 --arch resnet50
```

Available protocols:
* `1` : `AFLW2000` and `BIWI`
* `2` : `AFLW-test` subset
* `3` : `BIWI-test` subset

Available architectures:
* `resnet18`
* `resnet34`
* `resnet50`
* `resnet101`
* `resnet152`

## Bounding box labeling

Folder `data/` contains couple of JSON files with bounding box labeling.
These files have following format:
```
{
    "filename": [
        {"bbox": [x_min, y_min, x_max, y_max], "type"?: "train|test", ...},
        ...
    ]
}
```

## Citing

TBD
