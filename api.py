from flask import Flask, jsonify, request
import json
import numpy as np

app=Flask(__name__)
app.config['JSON_AS_ASCII'] = False
@app.route('/test',methods=['POST'])
def greeting():
    params=request.json
    print(params['test'])
    return jsonify({'token':"가"},{'token':"B"}),200


@app.route('/point',methods=['POST'])
def point():
    
    LEFT_EYE_INDEXES = [384, 385, 386, 387, 388, 390, 263, 362, 398, 466, 373, 374, 249, 380, 381, 382]
    RIGHT_EYE_INDEXES = [160, 33, 161, 163, 133, 7, 173, 144, 145, 246, 153, 154, 155, 157, 158, 159]
    MOUTH = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 95, 88, 178, 87, 14, 317, 402, 318, 324]

    hands=[]
    flag1=0
    flag2=0
    params=request.json
    if params['leftHand']:
        landmarks = params['leftHand']

        joint = np.zeros((21, 4))
        for j, lm in enumerate(landmarks):
            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

        v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]  # Parent joint
        v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                 :3]  # Child joint

        v = v2 - v1

        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

        angle = np.arccos(np.einsum('nt,nt->n',
                                    v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                    v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
        angle = np.degrees(angle)
         
        d = np.concatenate([joint.flatten(), angle])

        hands.extend(d)

    else:
        joint = np.zeros((21, 4))
        label=[0]*15
        d=np.concatenate([joint.flatten(),label])
        flag1=1
        hands.extend(d)

     # 오른손
    if params['rightHand']:
        landmarks = params['rightHand']

        joint2 = np.zeros((21, 4))
        for j, lm in enumerate(landmarks):
            joint2[j] = [lm.x, lm.y, lm.z, lm.visibility]

        v3 = joint2[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19],
                 :3]  # Parent joint
        v4 = joint2[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                 :3]  # Child joint
        v5 = v4 - v3

        v5 = v5 / np.linalg.norm(v5, axis=1)[:, np.newaxis]

        angle2 = np.arccos(np.einsum('nt,nt->n',
                                        v5[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                        v5[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
        angle2 = np.degrees(angle2)
         
         
        d2 = np.concatenate([joint2.flatten(), angle2])
        hands.extend(d2)

    else:
        joint2 = np.zeros((21, 4))
        label=[0]*15
        d2=np.concatenate([joint2.flatten(),label])
        flag2=1
        hands.extend(d2)

     # POSE
    if params['pose']:
        landmarks = params['pose']

        joint3 = np.zeros((6, 4))
        for j, lm in enumerate(landmarks):
            if j in [11,12,13,14,15,16]:
                if j==11: j=0
                if j==12: j=1
                if j==13: j=2
                if j==14: j=3
                if j==15: j=4
                if j==16: j=5
                 
                joint3[j] = [lm.x, lm.y, lm.z, lm.visibility]

        v6 = joint3[[1, 3, 0, 2], :3]
        v7 = joint3[[3, 5, 2, 4], :3]
        v8=v7-v6
        v8 = v8 / np.linalg.norm(v8, axis=1)[:, np.newaxis]


        angle3 = np.arccos(np.einsum('nt,nt->n',
                                     v8[[0, 2], :],
                                     v8[[1, 3],  :]))
        angle3 = np.degrees(angle3)
         
        #print("길이",len(angle3_label))
        d3 = np.concatenate([joint3.flatten(), angle3])
        #print(len(d3))
        hands.extend(d3)

    else:
        joint3 = np.zeros((6, 4))
        label=[0]*2
        d3=np.concatenate([joint3.flatten(),label])
        #print(len(d3))
        hands.extend(d3)


     # 얼굴
    if params['face']:
        landmarks = params['face']
         
        pointSize=len(LEFT_EYE_INDEXES)+len(RIGHT_EYE_INDEXES)+len(MOUTH)

        joint4=np.zeros((pointSize,4))
         
        for j, lm in enumerate(landmarks):
            if j in MOUTH:
                joint4[a] = [lm.x, lm.y, lm.z, lm.visibility]
                a+=1
            if j in LEFT_EYE_INDEXES:
                joint4[a] = [lm.x, lm.y, lm.z, lm.visibility]
                a+=1
            if j in RIGHT_EYE_INDEXES:
                joint4[a] = [lm.x, lm.y, lm.z, lm.visibility]
                a+=1
        #idx_label=[idx]*pointSize
        #d4=np.concatenate([joint4.flatten(),idx_label])
        d4=joint4.flatten()
        #print(len(d4))
        a=0
        hands.extend(d4)

    else:
        pointSize=len(LEFT_EYE_INDEXES)+len(RIGHT_EYE_INDEXES)+len(MOUTH)
        joint4=np.zeros((pointSize,4))
        d4=joint4.flatten()
        #print(len(d4))
        hands.extend(d4)


    return jsonify(hands),200

if __name__=="__main__":
    app.debug=True
    app.run()