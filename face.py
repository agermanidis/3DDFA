
class FaceModel:
    def __init__(self, image):
        self.image = image
        self.mode = 'cpu'
        self.dlib_landmark = True
        self.dlib_bbox = True #changed
        self.dump_obj = True
        self.paf_size = 3
        self.dump_paf = False
        self.dump_pncc = True
        self.dump_depth = True
        self.dump_pose = True
        self.dump_roi_box = False
        self.dump_pts = True
        self.dump_ply = True
        self.dump_vertex = False
        self.dump_res = True
        self.bbox_init = 'one'
        self.show_flg = True

    def sample(self, **kwargs):
        img_fp = 'samples/test1.jpg'
        # 2. load dlib model for face detection and landmark used for face cropping
        if self.dlib_landmark:
            dlib_landmark_model = 'models/shape_predictor_68_face_landmarks.dat'
            face_regressor = dlib.shape_predictor(dlib_landmark_model)
        if self.dlib_bbox:
            face_detector = dlib.get_frontal_face_detector()

        # 3. forward
        tri = sio.loadmat('visualize/tri.mat')['tri']
        transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
        #print(transform)
        if self.dlib_bbox:
            rects = face_detector(self.image, 1)
        else:
            rects = []

        pts_res = []
        Ps = []  # Camera matrix collection
        poses = []  # pose collection, [todo: validate it]
        vertices_lst = []  # store multiple face vertices
        ind = 0
        suffix = get_suffix(img_fp)
        for rect in rects:
            # whether use dlib landmark to crop image, if not, use only face bbox to calc roi bbox for cropping
            if self.dlib_landmark:
                # - use landmark for cropping
                pts = face_regressor(self.image, rect).parts()
                pts = np.array([[pt.x, pt.y] for pt in pts]).T
                roi_box = parse_roi_box_from_landmark(pts)
            else:
                # - use detected face bbox
                bbox = [rect.left(), rect.top(), rect.right(), rect.bottom()]
                roi_box = parse_roi_box_from_bbox(bbox)

            img = crop_img(self.image, roi_box)

            # forward: one step
            img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
            model_input = transform(img).unsqueeze(0)
            print(model_input)
            with torch.no_grad():
                
                if self.mode == 'gpu':
                    input = input.cuda()

                param = model(model_input)
                param = param.squeeze().cpu().numpy().flatten().astype(np.float32)


            # 68 pts
            pts68 = predict_68pts(param, roi_box)

            # two-step for more accurate bbox to crop face
            if self.bbox_init == 'two':
                roi_box = parse_roi_box_from_landmark(pts68)
                img_step2 = crop_img(img_ori, roi_box)
                img_step2 = cv2.resize(img_step2, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
                input = transform(img_step2).unsqueeze(0)
                with torch.no_grad():
                    if self.mode == 'gpu':
                        input = input.cuda()
                    param = model(input)
                    param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

                pts68 = predict_68pts(param, roi_box)

            pts_res.append(pts68)
            P, pose = parse_pose(param)
            Ps.append(P)
            poses.append(pose)

        
            # dense face 3d vertices
            if self.dump_ply or self.dump_vertex or self.dump_depth or self.dump_pncc or self.dump_obj:
                vertices = predict_dense(param, roi_box)
                vertices_lst.append(vertices)
            ind += 1
            

        pncc_feature = cpncc(self.image, vertices_lst, tri - 1)
        output = pncc_feature[:, :, ::-1]
        pilImg = transforms.ToPILImage()(np.uint8(output))
        return pilImg

        
