---
output:
  html_document: default
  pdf_document: default
---
# BMI3 Mini Project 
**Team No.227**

## Introduction

This folder contains the project report, source codes, trained CNN model and a command line executable file allow users to use our pre-trained model to classify their own skin lesion images. **The language version used for codes is python 3.7.**

- **add3_3e_4.h5:**  The trained CNN model
- **BMI3_project_report_Team_No.227:**  Project report
- **mini-project-final.py:**  Main source code, contains codes and discriptions of image preprocessing, model construction, accuracy evaluation & visaulization
- **picture extraction.py:**  Picture extraction source code, no need to peform in usual cases because it's not an ideal way to deal with imbalanced data
- **prediction_probability.py:**  User command line executable file, use this as a tool to generate your own image prediction output in a probability csv! For implementation details, please see below:

## Getting Started

1. Enter python environment 
```text
python3
```
2. Create a 'test' folder under your provided directory to hold the images.
```text
cd your/test/dir
mkdir test
```
3. Input command lines
```text
python3 prediction_probability.py -n your/output/filename -d your/test/dir -o your/output/dir
```

4. Get your prediction result! 

