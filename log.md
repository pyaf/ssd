CLS_THRESH: mAP


vast7oct/model10.pth
val: (8mins)
0.3 :  0.08007408126293993
0.35 :  0.07797133799171838
0.4 :  0.07367688923395443
0.45 :  0.06716647256728778
0.5 :  0.05941462862318841
0.55 :  0.05140802924430642

train: (53 mins)
0.3 :  0.08617854310842007
0.35 :  0.08284899437716291
0.4 :  0.07857509731833935
0.45 :  0.07289431949250313
0.5 :  0.06528997260669
0.55 :  0.055003964821222676

vast7oct/model20.pth
0.3 :  0.07800692287784686
0.35 :  0.0785439311594203
0.4 :  0.07598990683229817
0.45 :  0.07206101190476191
0.5 :  0.06872088509316772
0.55 :  0.061725219979296087  


# NOTE: vast/*.pth are overfitted model. Due to a bug, they were trained on whole of dataset so val set mAP is not actually "val set" mAP!

vast/model50.pth
0.3 :  0.170599605331263
0.35 :  0.1711034549689442
0.4 :  0.1705413755175984
0.45 :  0.17162509704968948
0.5 :  0.16991459627329195
0.55 :  0.16841841356107662


vast/model80.pth
0.3 :  0.1904349443581781
0.35 :  0.19019232013457557
0.4 :  0.19006292054865426
0.45 :  0.19054816899585925
0.5 :  0.19054816899585925
0.55 :  0.19019232013457557


model80.pth and model50.pth had bugs, overfitted

model80.pth > model50.pth on val set but on submision model80.pth (0.34) < model50.pth (0.37)