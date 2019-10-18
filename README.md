# Bachelor Projektarbeit
Basketball-Wurfanalyse mithilfe von Python, neuronalen Netzen sowie openCV. <br/>
Es soll ein vorher gefilmtes Video von der Software hinsichtlich des Wurfverhaltens des gefilmten Spieler analysiert werden. <br/>

## Setup ##
[Videos und Trainingsdaten](https://www.dropbox.com/sh/jru069x8v3w1gp3/AACdkQ-0Xbp_38oReUBgTlSUa?dl=0) herunterladen und jeweils im Skript 
an den benötigten Stellen referenzieren. <br/><br/>
Zum Ausführen: <br/>
``
python real_time_object_detection.py
``

## Wichtige Links ##
* [Motion Detection](https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/)          
* [Pedestrian Detection](https://www.pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/)                                     
* [How does detection work?](https://thedatafrog.com/human-detection-video/)                                                        
* [Object detection with YOLO](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/)                         
* [Realtime Object Detection](https://www.pyimagesearch.com/2017/09/18/real-time-object-detection-with-deep-learning-and-opencv/)   



## Status & Tasks ##
- [x] Spieler mithilfe von neuronalem Netz erkennen
- [x] Suchframe automatisch generieren
- [x] Marker in Suchframe tracken
- [x] Arm-Markierungen labeln
- [ ] Verbindungen zwischen Arm-Markierungen zeichnen
- [ ] Winkel und andere Werte kalkulieren
- [ ] Abgleich mit z.B. "perfektem Wurf" von Nowitzki
- [ ] Nutzer hat die Möglichkeit die getrackte Farbe anzugeben


## Probleme ##
* Arm-Markierungen konstant tracken, um Linien zu zeichnen. Sortierung nach x-Wert klappt perfekt, 
  nur ist das Handgelenk irgendwann im Wurfablauf "zwischen" Ellenbogen und Schulter.
* Die vielen Array-Berechnungen schlagen sich in der Performance nieder.