# Bachelor Projektarbeit
Basketball-Wurfanalyse mithilfe von Python, neuronalen Netzen sowie openCV. <br/>
Es soll ein vorher gefilmtes Video von der Software hinsichtlich des Wurfverhaltens, mit Fokus auf den Winkel des Wurfarmes, des gefilmten Spieler analysiert werden. <br/>

## Setup ##
[Videos und Trainingsdaten](https://www.dropbox.com/sh/jru069x8v3w1gp3/AACdkQ-0Xbp_38oReUBgTlSUa?dl=0) herunterladen und jeweils im Skript 
an den benötigten Stellen referenzieren bzw. eigene Videos nutzen. <br/><br/>
Zum Ausführen: <br/>
``
python real_time_object_detection.py
``

## Links ##
* [Motion Detection](https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/)          
* [Pedestrian Detection](https://www.pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/)                                     
* [How does detection work?](https://thedatafrog.com/human-detection-video/)                                                        
* [Object detection with YOLO](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/)                         
* [Realtime Object Detection](https://www.pyimagesearch.com/2017/09/18/real-time-object-detection-with-deep-learning-and-opencv/)   
* [Shooting a Basketball](https://www.breakthroughbasketball.com/fundamentals/shooting-arm-wrist-angle.html)
* [Scientific paper about shooting a basketball](https://pdf.sciencedirectassets.com/278653/1-s2.0-S1877705816X00161/1-s2.0-S187770581630649X/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEHIaCXVzLWVhc3QtMSJIMEYCIQDJX9pcIbL3fByndQLSm%2Bd%2FyXCRNMdHs%2F9LWISZiczbrgIhAJHczcuN%2FLvgOkAC%2FbLU7qb17X%2FF8LxSuN8SOeXPvvghKtkCCIv%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQAhoMMDU5MDAzNTQ2ODY1IgxlM3e%2Bumtz5sm2RTMqrQKdZvvYXCp4D0jkH%2B77JZ30mCsUyqYk%2FRyRoyxNhLJNDyHCR%2BMoh1OPfcjdW3pVoXlXt6qeBlU6TGXgbToHeDn9yS1u%2Brs1liQ%2Bq51rtSTQPc5XaVzu9UEzorIlAuMwSPz2hay%2FMvw0qLWdFSPqgc1%2Ft7zI%2BkmLGRI8sLPQEiF1ga8GupiuLI%2B%2FC%2B8QJ9Jv%2BTHkczf%2BMtljMkAU%2BN0ZRG1IFs%2FaggHYmEodUDiWNoQFxYD%2BDnl25HhD4hgWoD9v4oB%2BUgZEeefQ1ZiS6KF5lfJS9yNHaAW01AB3nbUPbwI%2FewCugpFhDYGcTvObTF1l5xR7dvORoAOrZMv8nXDK27N2iQu7H0fDiOH8%2B9bC5TPQLUukqzN9zwW2AQWDo8u1ybmnBMbA3Bb41UUy9UTeMI3Xj%2B4FOs4C9xoKpuWQ%2FDVlakYv2k1dyJy5eAb5L%2FayFSWLe3cR0u3r3uRph3d4IUSTLXXGFaLUtDyrLCkSqfBLc9gfyoDbR3AdqRzzzZu5SlZv9dTFdu3wmeKElt4AixCnq8%2B0ucwZ6j5zaJnd6oALU7weh9RYGJXX%2BUuiM%2F8uNL9kuBt1lXthI1k938LoRypBC4vRoURLW%2Bob0yFMMMIDCI2Or8%2FLOrXcxZKtMNz0lz0%2FJHab%2FrEUzDzMrMHzkZdC8H2xM1Ex3wmIlWRhvotbM3mrIE%2B3cNIUaURrWAwBnK%2BoJxKmJDMIX%2BMj%2B5QraGj9M8rCRB%2FN6jjFSzrvTKRjnFLPVjaWX3L6nVFZc9tfDWoJVRRLJ7mdW3L3LlugKBz%2FXeW0B7YfdwnVIAMVW2qzENxCm%2BQuD%2Bf277G%2FarFbq27cST2qC5lRgsAczYzOHr3EubarbA%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20191107T102549Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY5NOVDOZR%2F20191107%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=d33a73f555f8559221cda275ff3912add9636beb01d102dfe44849586111184a&hash=4636b184042d2a89c12364838dfe85d66e9714b459fdf9e51c9ba32cef2c3597&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S187770581630649X&tid=spdf-52b2be62-6415-4127-a0e0-111f420162ba&sid=d902a5db4c0ff84008590294dbedba901c8fgxrqb&type=client)
* [DensePose](https://arxiv.org/pdf/1802.00434.pdf)

## Status & Tasks ##
- [x] Spieler mithilfe von neuronalem Netz erkennen
- [x] Suchframe automatisch generieren
- [x] Marker in Suchframe tracken
- [x] Arm-Markierungen labeln
- [x] Verbindungen zwischen Arm-Markierungen zeichnen
- [x] Winkel in der Armbeuge kalkulieren
- [x] Treffer erkennen
- [x] Nur bei Wurf den Winkel berechnen
- [x] Minimal- und Maximalwinkel für derzeitigen Wurf tracken
- [ ] Speichern, wann am erfolgreichsten geworfen wurde
- [ ] Statistik nach Programmdurchlauf ausgeben

### Nice-to-have ###
- [ ] Performance verbessern (z.B. skalieren)
- [ ] Tracking verbessern.
- [ ] Nutzer hat die Möglichkeit die getrackte Farbe anzugeben


## Probleme ##
* Arm-Markierungen konstant tracken, um Linien zu zeichnen. Sortierung nach x-Wert klappt perfekt, 
  nur ist das Handgelenk irgendwann im Wurfablauf "zwischen" Ellenbogen und Schulter. (SOLVED)
* Die vielen Array-Berechnungen schlagen sich in der Performance nieder.
* Markierungen dauerhaft perfekt tracken.
* Richtige Reihenfolge des Marker-Arrays immer gewährleisten, besonders kurz vor Beginn eines Wurfs. 
  Es ist nicht möglich, nur anhand der Koordinaten der Marker immer perfekt zu bestimmen, ob es sich
  um Hand, Ellenbogen oder Schulter handelt.
* Durch das nicht immer perfekte ist der berechnete Min/Max-Winkel nicht immer korrekt.


## Milestones ##
* Person im Bild wird immer erkannt.
* Fixpunkte am Arm des Werfers (Schulter, Ellenbogen, Handgelenk) werden richtig getrackt.
* Vewendung einer anderen Videoquelle funktioniert direkt ohne etwas anpassen zu müssen.
* Winkel in der Armbeuge tracken.
* Score tracken über Markierung am Netz.
