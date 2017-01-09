# coding: utf-8

from dqn_pendulum import DQNAgent, pendulumEnvironment, simulator
from chainer import serializers
from make_gif import log2gif

import threading
from PyQt4.QtGui import QCheckBox, QFont, QApplication, QLabel, QMovie, QPainter, QWidget, QGridLayout
import sys
import time
import numpy as np

GIF_INTERVAL = 100


# Result window
class UI(QWidget):

    def __init__(self):
        super(UI, self).__init__()
        self.initUI()
        self.count = 0

    # initialize window
    def initUI(self):
        self.font = QFont('Calibri')
        self.setFont(self.font)
        self.setGeometry(50, 50, 270, 256)
        self.setWindowTitle("DQN")

        # When showSignal's state is changed, update gif
        global showSignal
        showSignal = QCheckBox('', self)
        showSignal.setHidden(True)
        showSignal.stateChanged.connect(self.showGif)

        self.movie = QLabel()  # Gif
        self.label = QLabel()  # Label

        self.layoutMain = QGridLayout()
        self.layoutMain.addWidget(self.movie, 0, 0)
        self.layoutMain.addWidget(self.label, 1, 0)
        self.layoutMain.setRowStretch(0, 12)
        self.setLayout(self.layoutMain)

    # update gif
    def setMovie(self, gifName):
        m = QMovie(gifName)
        m.start()
        self.movie.setMovie(m)

    # Update label and gif
    def showGif(self):
        gifName = "gif/" + str(self.count * GIF_INTERVAL) + ".gif"
        self.label.setText(gifName)
        self.setMovie(gifName)
        self.count += 1

    # When window is closed, other threads finish
    def closeEvent(self, event):
        global flagEnd
        flagEnd = True


# Thread for train
def threadTrain():
    # signals for other threads
    global makeSignal
    global flagEnd
    global globalLog
    flagEnd = False

    agent=DQNAgent()
    env=pendulumEnvironment()
    sim=simulator(env,agent)

    test_highscore=0

    fw=open("log.csv","w")

    for i in range(30000):
        if flagEnd:
            break
        total_reward=sim.run(train=True, movie=False)

        if i%1000 ==0:
            serializers.save_npz('model/%06d.model'%i, agent.model)

        if i%GIF_INTERVAL == 0:
            # [add]
            # total_reward=sim.run(train=False, movie=True)
            total_reward, globalLog = sim.run(train=False, movie=True, enableLog=True)
            sim.log = []  # clear log
            makeSignal = True  # makeSignal is sent to "threadMakeGif"

            if test_highscore<total_reward:
                # print "highscore!",
                print("highscore!")
                serializers.save_npz('model/%06d_hs.model'%i, agent.model)
                test_highscore=total_reward
            # print i,
            # print total_reward,
            # print "epsilon:%2.2e" % agent.get_epsilon(),
            # print "loss:%2.2e" % agent.loss,
            aw=agent.total_reward_award
            # self.ResultUpDown.setText(str("{0:.1f}".format(np.mean(settling) + 0.9)))
            print(i, "{0:.3f}".format(total_reward),
                  "epsilon:", "{0:.3f}".format(agent.get_epsilon()),
                  "loss:", "{0:.3f}".format(float(agent.loss)),
                  "min:", "{0:.3f}".format(np.min(aw)),
                  "max:", "{0:.3f}".format(np.max(aw)))

            out="%d,%d,%2.2e,%2.2e,%d,%d\n" % (i,total_reward,agent.get_epsilon(),agent.loss, np.min(aw),np.max(aw))
            fw.write(out)
            fw.flush()
    fw.close

    # If this thread finished, other threads finish
    flagEnd = True


# Thread for making gif
def threadMakeGif():
    # signals for other threads
    global makeSignal
    global showSignal
    global globalLog
    global flagEnd
    flagEnd = False
    makeSignal = False
    count = 0

    while True:
        # If other threads finished, exit loop
        if flagEnd:
            break
        # If makeSignal become True, make gif and showSignal become True
        # (makeSignal comes from "threadTrain", showSignal is sent to "threadShowGif")
        elif makeSignal:
            gifName = "gif/" + str(count * GIF_INTERVAL) + ".gif"
            log2gif(globalLog, gifName, "")

            count += 1
            makeSignal = False
            if showSignal.isChecked():
                showSignal.setChecked(False)
            else:
                showSignal.setChecked(True)
        else:
            time.sleep(1)

    # If this thread finished, other threads finish
    flagEnd = True


if __name__ == '__main__':
    th_train = threading.Thread(target=threadTrain)
    th_makeGif = threading.Thread(target=threadMakeGif)
    th_train.start()
    th_makeGif.start()

    # show result window
    app = QApplication(sys.argv)
    ui = UI()
    ui.show()
    sys.exit(app.exec_())
