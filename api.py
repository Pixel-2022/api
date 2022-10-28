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
    flag1=0
    flag2=0
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
            
            d = np.concatenate([joint.flatten(), angle])

            hands.extend(d)
    except:  
        joint = np.zeros((21, 4))
        label=[0]*15
        d=np.concatenate([joint.flatten(),label])
        flag1=1
        hands.extend(d)
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
            
            
            d2 = np.concatenate([joint2.flatten(), angle2])
            hands.extend(d2)

    except:
        joint2 = np.zeros((21, 4))
        label=[0]*15
        d2=np.concatenate([joint2.flatten(),label])
        flag2=1
        hands.extend(d2)
        pass

    # POSE
    try:
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
                    
                    joint3[j] = [lm[0], lm[1], lm[2], 0]

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

    except:
        joint3 = np.zeros((6, 4))
        label=[0]*2
        d3=np.concatenate([joint3.flatten(),label])
        #print(len(d3))
        hands.extend(d3)
        pass


    # 얼굴
    try:
        if params['face']:
            landmarks = params['face']
            
            pointSize=len(LEFT_EYE_INDEXES)+len(RIGHT_EYE_INDEXES)+len(MOUTH)
            
            joint4 = np.zeros((pointSize, 4))
            for j, lm in enumerate(landmarks):
                if j in [46,53,52,65,55,276,283,282,295,285, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 95, 88, 178, 87, 14, 317, 402, 318, 324,
                33,246,161,160,159,158,157,173,133, 7,163,144,145,153,154,155, 362,398,384,385,386,387,388,466,263, 382,381,380,374,373,390,249]:
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
                    if j==191: x=11
                    if j==80: x=12
                    if j==81: x=13
                    if j==82: x=14
                    if j==13: x=15
                    if j==312: x=16
                    if j==311: x=17
                    if j==310: x=18
                    if j==415: x=19
                    if j==308: x=20

                    if j==95: x=21
                    if j==88: x=22
                    if j==178: x=23
                    if j==87: x=24
                    if j==14: x=25
                    if j==317: x=26
                    if j==402: x=27
                    if j==318: x=28
                    if j==324: x=29
                    
                    if j==33: x=30
                    if j==246: x=31
                    if j==161: x=32
                    if j==160: x=33
                    if j==159: x=34
                    if j==158: x=35
                    if j==157: x=36
                    if j==173: x=37
                    if j==133: x=38
                    if j==7: x=39
                    if j==163: x=40
                    if j==144: x=41
                    if j==145: x=42
                    if j==153: x=43
                    if j==154: x=44
                    if j==155: x=45

                    if j==362: x=46
                    if j==398: x=47
                    if j==384: x=48
                    if j==385: x=49
                    if j==386: x=50
                    if j==387: x=51
                    if j==388: x=52
                    if j==466: x=53
                    if j==263: x=54
                    if j==382: x=55
                    if j==381: x=56
                    if j==380: x=57
                    if j==374: x=58
                    if j==373: x=59
                    if j==390: x=60
                    if j==249: x=61
                    
                    joint4[x] = [lm[0], lm[1], lm[2], 0]
            
                                                                                                                                                                 #35                              43                              51                              59             
            v_face1 = joint4[[0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 10, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 30, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 46, 55, 56, 57, 58, 59, 60, 61], :3]
            v_face2 = joint4[[1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 20, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 38, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 54], :3]
            v_face3 =v_face2-v_face1
            v_face3 = v_face3 / np.linalg.norm(v_face3, axis=1)[:, np.newaxis]

            angle4 = np.arccos(np.einsum('nt,nt->n',
                                        v_face3[[0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58], :],
                                        v_face3[[1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59],  :]))
            angle4 = np.degrees(angle4)
            
            d4=np.concatenate([joint4.flatten(), angle4])

            a=0
            hands.extend(d4)

    except:
        pointSize=len(LEFT_EYE_INDEXES)+len(RIGHT_EYE_INDEXES)+len(MOUTH)
        joint4=np.zeros((pointSize,4))
        label=[0]*52
        d4=np.concatenate([joint4.flatten(),label])
        #print(len(d4))
        hands.extend(d4)
        pass

    print(np.array(hands).shape)
    return jsonify(hands),200


if __name__=="__main__":
    app.debug=True
    app.run()
app.run(host="0.0.0.0", port="5000", debug=True)
