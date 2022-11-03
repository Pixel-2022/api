from flask import Flask, jsonify, request
import json
from importlib_metadata import NullFinder
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
    
    RIGHT_EYE_INDEXES = [46,53,52,65,55,33, 246,161,160,159,158,157,173,133, 7,163,144,145,153,154,155]
    LEFT_EYE_INDEXES = [276,283,282,295,285, 362,398,384,385,386,387,388,466,263, 382,381,380,374,373,390,249]
    MOUTH = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 95, 88, 178, 87, 14, 317, 402, 318, 324]

    a=0
    hands=[]
    joint_concat = np.zeros((6, 4))

    data_left = 0
    data_right = 0
    params=request.json
    try:
        if params['leftHand']:
            landmarks = params['leftHand']
            
            joint = np.zeros((21, 4))
            for j, lm in enumerate(landmarks):
                joint[j] = [lm[0], lm[1], lm[2], 0]

            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]  # Parent joint
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                    :3]  # Child joint

            v = v2 - v1

            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            angle = np.arccos(np.einsum('nt,nt->n',
                                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
            angle = np.degrees(angle)

            joint_concat[0] = joint[9]

            

            hands.extend(angle)
    except:  
        data_left=1
        joint = np.zeros((21, 4))
        joint_concat[0] = joint[9]
        label=[0]*15
        
        
        hands.extend(label)
        pass

# 오른손
    try:
        if params['rightHand']:
            landmarks = params['rightHand']

            joint2 = np.zeros((21, 4))
            for j, lm in enumerate(landmarks):
                joint2[j] = [lm[0], lm[1], lm[2], 0]

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
            
            joint_concat[1] = joint2[9]
            
            hands.extend(angle2)

    except:
        joint2 = np.zeros((21, 4))
        label=[0]*15
        joint_concat[1] = joint2[9]
        data_right=1
        hands.extend(label)
        pass

    # POSE
    try:
        if params['pose']:
            landmarks = params['pose']

            joint3 = np.zeros((8, 4))
            for j, lm in enumerate(landmarks):
                if j in [11,12,13,14,15,16]:
                    if j==11: j=0
                    if j==12: j=1
                    if j==13: j=2
                    if j==14: j=3
                    if j==15: j=4
                    if j==16: j=5
                    
                    joint3[j] = [lm[0], lm[1], lm[2], 0]

            joint3[6]=joint3[0]
            joint3[6][1]=joint3[6][1]+0.05
            joint3[7]=joint3[1]
            joint3[7][1]=joint3[7][1]+0.05

            v6 = joint3[[1, 3, 0, 2, 7, 6], :3]
            v7 = joint3[[3, 5, 2, 4, 1, 0], :3]
            v8=v7-v6
            v8 = v8 / np.linalg.norm(v8, axis=1)[:, np.newaxis]


            angle3 = np.arccos(np.einsum('nt,nt->n',
                                        v8[[0, 2, 4,5], :],
                                        v8[[1, 3, 0,2],  :]))
            angle3 = np.degrees(angle3)
            
            #print("길이",len(angle3_label))

            #print(len(d3))
            
            joint_concat[2]=joint3[2]
            joint_concat[3]=joint3[3]
            joint_concat[4]=joint3[4]
            joint_concat[5]=joint3[5]
            
            hands.extend(angle3)

    except:
        joint3 = np.zeros((8, 4))
        joint_concat[2]=joint3[2]
        joint_concat[3]=joint3[3]
        joint_concat[4]=joint3[4]
        joint_concat[5]=joint3[5]
        label=[0]*4

        #print(len(d3))
        hands.extend(label)
        pass


    # 얼굴
    try:
        if params['face']:
            landmarks = params['face']
            
            pointSize=len(LEFT_EYE_INDEXES)+len(RIGHT_EYE_INDEXES)+len(MOUTH)
            
            joint4 = np.zeros((pointSize, 4))
            for j, lm in enumerate(landmarks):
                if j in [46,53,52,65,55,276,283,282,295,285, 78, 81, 13, 311, 308, 178, 14, 402, 33,159,133,145, 362,386,263, 374]:
                    if j==46: x=0
                    if j==53: x=1
                    if j==52: x=2
                    if j==65: x=3
                    if j==55: x=4

                    if j==276: x=5
                    if j==283: x=6
                    if j==282: x=7
                    if j==295: x=8
                    if j==285: x=9

                    if j==78: x=10
                    if j==81: x=11
                    if j==13: x=12
                    if j==311: x=13
                    if j==308: x=14

                    if j==178: x=15
                    if j==14: x=16
                    if j==402: x=17
                    
                    if j==33: x=18
                    if j==159: x=19
                    if j==133: x=20
                    if j==145: x=21

                    if j==362: x=22
                    if j==386: x=23
                    if j==263: x=24
                    if j==374: x=25

                    joint4[x] = [lm[0], lm[1], lm[2], 0]
            
       
            v_face1 = joint4[[0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 17, 16, 15, 18, 19, 20, 21, 22, 23, 24, 25], :3]
            v_face2 = joint4[[1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 17, 16, 15, 10, 19, 20, 21, 18, 23, 24, 25, 22], :3]
            v_face3 =v_face2-v_face1
            v_face3 = v_face3 / np.linalg.norm(v_face3, axis=1)[:, np.newaxis]

            angle4 = np.arccos(np.einsum('nt,nt->n',
                    v_face3[[0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12,13,14,15, 16,17,18,19, 20,21,22,23], :],
                    v_face3[[1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13,14,15,8, 17,18,19,16, 21,22,23,20],  :]))
            angle4 = np.degrees(angle4)
            


            a=0
            hands.extend(angle4)

    except:
        pointSize=len(LEFT_EYE_INDEXES)+len(RIGHT_EYE_INDEXES)+len(MOUTH)
        joint4=np.zeros((pointSize,4))
        label=[0]*22

        #print(len(d4))
        hands.extend(label)
        pass


    v9_concat = joint_concat[[3,5,2,4], :3]
    v10_concat = joint_concat[[5,1,4,0], :3]
    v11_concat = v10_concat-v9_concat
    v11_concat=v11_concat/np.linalg.norm(v11_concat, axis=1)[:, np.newaxis]
    angle5=np.arccos(np.einsum('nt,nt->n',
                                        v11_concat[[0, 2], :],
                                        v11_concat[[1, 3], :]))
    angle5 = np.degrees(angle5)

    
    print(joint_concat)
    print(angle5)
    if data_left==1:
        angle5[0]=180
    if data_right==1:
        angle5[1]=180
    if angle5==[nan, nan]:
        angle5=[180,180]
    hands.extend(angle5)
    print(hands)

    return jsonify(hands),200


if __name__=="__main__":
    app.debug=True
    app.run()
app.run(host="0.0.0.0", port="5000", debug=True)
