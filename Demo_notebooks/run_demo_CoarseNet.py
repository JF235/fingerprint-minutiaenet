from __future__ import absolute_import
from __future__ import division
import matplotlib
matplotlib.use('Agg')
import os
if not os.path.exists('../output_old'): os.makedirs('../output_old')

import sys, os
sys.path.append(os.path.realpath('../CoarseNet'))

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ['KERAS_BACKEND'] = 'tensorflow'


from keras import backend as K

from MinutiaeNet_utils import *
from CoarseNet_utils import *
from CoarseNet_model import *
import argparse


config = K.tf.ConfigProto(gpu_options=K.tf.GPUOptions(allow_growth=True))
sess = K.tf.Session(config=config)
K.set_session(sess)




# Prepare dataset for testing. 
inference_set = ['../Dataset/CoarseNet_test/',]

CoarseNet_path = '../Models/CoarseNet.h5'

output_dir = '../output_old'

FineNet_path = '../Models/FineNet.h5'

logging = init_log(output_dir)

# If use FineNet to refine, set into True
isHavingFineNet = False

for i, deploy_set in enumerate(inference_set):
    set_name = deploy_set.split('/')[-2]

    # Read image and GT
    img_name, folder_name, img_size = get_maximum_img_size_and_names(deploy_set)

    pass # mkdir(output_dir + ...

    logging.info("Predicting \"%s\":" % (set_name))


    main_net_model = CoarseNetmodel((None, None, 1), CoarseNet_path, mode='deploy')

    # ====== Load FineNet to verify
    if isHavingFineNet == True:
        model_FineNet = FineNetmodel(num_classes=2,
                             pretrained_path=FineNet_path,
                             input_shape=(224,224,3))

        model_FineNet.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=0),
                      metrics=['accuracy'])

    for i in xrange(0, len(img_name)):
        
        logging.info("\"%s\" %d / %d: %s" % (set_name, i + 1, len(img_name), img_name[i]))

        image = misc.imread(deploy_set + 'img_files/' + img_name[i] + '.bmp', mode='L')# / 255.0

        img_size = image.shape
        img_size = np.array(img_size, dtype=np.int32) // 8 * 8
        image = image[:img_size[0], :img_size[1]]

        original_image = image.copy()

        # Generate OF
        texture_img = FastEnhanceTexture(image, sigma=2.5, show=False)
        dir_map, fre_map = get_maps_STFT(texture_img, patch_size=64, block_size=16, preprocess=True)
        
        image = np.reshape(image, [1, image.shape[0], image.shape[1], 1])

        enh_img, enh_img_imag, enhance_img, ori_out_1, ori_out_2, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out \
            = main_net_model.predict(image)

        # Use for output mask
        round_seg = np.round(np.squeeze(seg_out))
        seg_out = 1 - round_seg
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        seg_out = cv2.morphologyEx(seg_out, cv2.MORPH_CLOSE, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        seg_out = cv2.morphologyEx(seg_out, cv2.MORPH_OPEN, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        seg_out = cv2.dilate(seg_out, kernel)

        #========== Adaptive threshold ==================
        final_minutiae_score_threashold = 0.45
        early_minutiae_thres = final_minutiae_score_threashold + 0.05



        # In cases of small amount of minutiae given, try adaptive threshold
        while final_minutiae_score_threashold >= 0:
            mnt = label2mnt(np.squeeze(mnt_s_out) * np.round(np.squeeze(seg_out)), mnt_w_out, mnt_h_out, mnt_o_out,
                            thresh=early_minutiae_thres)

            mnt_nms_1 = py_cpu_nms(mnt, 0.5)
            mnt_nms_2 = nms(mnt)
            # Make sure good result is given
            if mnt_nms_1.shape[0] > 4 and mnt_nms_2.shape[0] > 4:
                break
            else:
                final_minutiae_score_threashold = final_minutiae_score_threashold - 0.05
                early_minutiae_thres = early_minutiae_thres - 0.05


        mnt_nms = fuse_nms(mnt_nms_1, mnt_nms_2)

        mnt_nms = mnt_nms[mnt_nms[:, 3] > early_minutiae_thres, :]
        mnt_refined = []

        if isHavingFineNet == True:
            # ======= Verify using FineNet ============
            patch_minu_radio = 22
            if FineNet_path != None:
                for idx_minu in range(mnt_nms.shape[0]):
                    try:
                        # Extract patch from image
                        x_begin = int(mnt_nms[idx_minu, 1]) - patch_minu_radio
                        y_begin = int(mnt_nms[idx_minu, 0]) - patch_minu_radio
                        patch_minu = original_image[x_begin:x_begin + 2 * patch_minu_radio,
                                     y_begin:y_begin + 2 * patch_minu_radio]

                        patch_minu = cv2.resize(patch_minu, dsize=(224, 224), interpolation=cv2.INTER_NEAREST)

                        ret = np.empty((patch_minu.shape[0], patch_minu.shape[1], 3), dtype=np.uint8)
                        ret[:, :, 0] = patch_minu
                        ret[:, :, 1] = patch_minu
                        ret[:, :, 2] = patch_minu
                        patch_minu = ret
                        patch_minu = np.expand_dims(patch_minu, axis=0)

                        # # Can use class as hard decision
                        # # 0: minu  1: non-minu
                        # [class_Minutiae] = np.argmax(model_FineNet.predict(patch_minu), axis=1)
                        #
                        # if class_Minutiae == 0:
                        #     mnt_refined.append(mnt_nms[idx_minu,:])

                        # Use soft decision: merge FineNet score with CoarseNet score
                        [isMinutiaeProb] = model_FineNet.predict(patch_minu)
                        isMinutiaeProb = isMinutiaeProb[0]
                        # print isMinutiaeProb
                        tmp_mnt = mnt_nms[idx_minu, :].copy()
                        tmp_mnt[3] = (4*tmp_mnt[3] + isMinutiaeProb) / 5
                        mnt_refined.append(tmp_mnt)

                    except:
                        mnt_refined.append(mnt_nms[idx_minu, :])
        else:
            mnt_refined = mnt_nms

        mnt_nms_backup = mnt_nms.copy()
        mnt_nms = np.array(mnt_refined)

        if mnt_nms.shape[0] > 0:
            mnt_nms = mnt_nms[mnt_nms[:, 3] > final_minutiae_score_threashold, :]
        
        final_mask = ndimage.zoom(np.round(np.squeeze(seg_out)), [8, 8], order=0)
        # Show the orientation
        show_orientation_field(original_image, dir_map + np.pi, mask=final_mask, fname="%s/%s_OF.jpg" % (output_dir, img_name[i]))

        fuse_minu_orientation(dir_map, mnt_nms, mode=3)

        time_afterpost = time()
        mnt_writer(mnt_nms, img_name[i], img_size, "%s/%s.mnt"%(output_dir, img_name[i]))
        draw_minutiae(original_image, mnt_nms, "%s/%s_minu.jpg"%(output_dir, img_name[i]),saveimage=True)

        misc.imsave("%s/%s_seg.jpg" % (output_dir, img_name[i]), final_mask)



