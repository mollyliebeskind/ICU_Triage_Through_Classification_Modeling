# ICU_Triage_Through_Classification_Modeling
Using classification modeling to identify and triage high risk patients in the ICU

## Objective:
Develop a classification model to identify high-risk patients in the ICU and assign a triage tag.

## Data
Dataset was developed as part of MIT's GOSSIS community initiative, with privacy certification from the Harvard Privacy Lab. https://www.kaggle.com/c/widsdatathon2020/data

## Triage Protocal

Green: (wait) are reserved for the "walking wounded" who will need medical care at some point, after more critical injuries have been treated.
Yellow: (observation) for those who require observation (and possible later re-triage). Their condition is stable for the moment and, they are not in immediate danger of death. These victims will still need hospital care and would be treated immediately under normal circumstances.
Red: (immediate) are used to label those who cannot survive without immediate treatment but who have a chance of survival.
Black: (expectant) are used for the deceased and for those whose injuries are so extensive that they will not be able to survive given the care that is available. Note: this model will not identify black tags. Within the red group, physicians should assign additional tagging if needed.

## Metrics
The goal is to flag anyone entering the ICU who is at risk. With that, recall is the main metric although AUC-ROC and precision will be used as secondary considerations.
