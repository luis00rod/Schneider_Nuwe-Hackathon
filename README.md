# Schneider_Nuwe-Hackathon
Este algoritmo predice el tipo de polucion, la data utilizada es proporcionada por la Union Europea a traves de la plataforma Nuwe.

Se cargo la data de diferentes archivos y se proceso la data y limpio la data. 

El modelo fue entrenado utilizando randomforest

Las variables predictoras son las siguientes:

countryName: Country in which the facility is located

EPRETRSectorCode: Code of the sector in which the company specialises

eptrSectorName: Name of the sector in which it specialises

EPRTRAnnexIMainActivityCode: Code of the specialisation within the sector in which they operate

EPRTRAnnexIMainActivityLabel: Specialisation within the sector in which they operate

FacilityInspireID: Building identifier

facilityName: Name of the building in which the activity takes place

City: City in which the facility is located

CITY ID: ID to confirm location

targetRelease: Type of polluter to study

DAY: Day on which the report is made

MONTH: Reporting month

reportingYear: Reporting year

CONTINENT: Continent on which the company is located

max_wind_speed: Maximum wind speed

avg_wind_speed: Average wind speed

min_wind_speed: Minimum wind speed

max_temp: Maximum temperature

avg_temp: Average temperature

min_temp: Minimum temperature

DAYS WITH FOG: Total days of the month recorded in the area

REPORTER NAME: Reporter's name

Variable a predecir es:

pollutant: Type of pollutant emitted. In order to follow the same standard, you must encode this variables as follows:

Este compaticion fue realizada por:

Victor Rodriguez

Luis Rodriguez




