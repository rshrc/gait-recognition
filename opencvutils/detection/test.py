from detection import MotionDetection, PedestrianDetection, FaceDetection


def motion_test():
    motion_obj = MotionDetection()
    motion_obj.run()


def face_test():
    face_obj = FaceDetection()
    face_obj.run()


def ped_test():
    ped_obj = PedestrianDetection()
    ped_obj.run()


#motion_test()#face_test()
ped_test()
