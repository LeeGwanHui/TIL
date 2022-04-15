class yoloLoss(nn.Module):
    def __init__(self,l_coord,l_noobj):
        super(yoloLoss,self).__init__()
        self.l_coord = l_coord
        self.l_noobj = l_noobj
    
    def compute_prob_error(self, pred, target): ## classification error
        contain = target[:,:,:,4] > 0 # [Batch_size, 7, 7, 1]
        contain = contain.unsqueeze(3).expand_as(target) # [Batch_size, 7, 7, 30]
        
        ################
        ### Fill out ###
        ################
        contain_pred = pred[contain].view(-1, 30)          # [n_coord, 30]
                                                           # n_coord: object가 있는 cell의 수
        contain_target = target[contain].view(-1, 30)      # [n_coord, 30]
                                                           # n_coord: object가 있는 cell의 수
        class_pred = contain_pred[:, 5*2:] 
        class_target = contain_target[:, 5*2:]
        prob_loss = F.mse_loss(class_pred, class_target, reduction='sum')

        return prob_loss
    
    # 위의 loss로 인해서 우리는 object가 존재하는 cell에 대해서 class probability를 학습 시킬 수 있다. 
    
    def not_contain_obj_error(self, pred, target): ## Error where no object appears in cell
        not_contain = target[:,:,:,4] == 0 # [Batch_size, 7, 7, 1]
        not_contain = not_contain.unsqueeze(3).expand_as(target) # [Batch_size, 7, 7, 30]

        not_obj_pred = pred[not_contain].view(-1,30) # [n_noobj, 30] n_noobj cell에 object가 없는 수
        not_obj_target = target[not_contain].view(-1,30) # [n_noobj, 30] n_noobj cell에 object가 없는 수
        
        not_obj_conf_mask = torch.cuda.ByteTensor(not_obj_pred.size()).fill_(0) # [n_noobj, 30]
        for b in range(2):
            not_obj_conf_mask[:, 4 + b*5] = 1 # not_obj_conf_mask[:, 4] = 1; not_obj_conf_mask[:, 9] = 1
        
        not_obj_pred_conf = not_obj_pred[not_obj_conf_mask]       #  object가 없는 confidence score만 추출 
                                                                  # [n_noobj, 2=len([conf1, conf2])]
        not_obj_target_conf = not_obj_target[not_obj_conf_mask]   #  object가 없는 confidence score만 추출 
                                                                  # [n_noobj, 2=len([conf1, conf2])]
        noobj_loss1 = F.mse_loss(not_obj_pred_conf, not_obj_target_conf, reduction='sum')
        
        return noobj_loss1
    
    # 이 loss를 가지고 우리는 background로 추출하는 confidence score을 0으로 만들도록 학습시킬 수 있다.
    
    
    def contain_obj_error(self, pred, target): ##
        contain = target[:,:,:,4] > 0 # [Batch_size, 7, 7, 1]
        contain = contain.unsqueeze(3).expand_as(target) # [Batch_size, 7, 7, 30]
        
        contain_pred = pred[contain].contiguous().view(-1,30) # Only cell which contains the object are remained
        box_pred = contain_pred[:,:10].contiguous().view(-1,5) # Reshaping [n_coord x 2, 5=len([x, y, w, h, conf])]
        
        contain_target = target[contain].contiguous().view(-1,30) # Only cell which contains the object are remained
        box_target = contain_target[:,:10].contiguous().view(-1,5)# Reshaping [n_coord x 2, 5=len([x, y, w, h, conf])]
        
        # Note that, We assign one predictor to be "responsible" for predicting 
        # an object based on which prediction has the highest current IOU with the ground truth
        # Then, other prediction is "not responsible" for predicting an object
        # That component is included in "contain_noobj_mask variables
        
        contain_obj_mask = torch.cuda.ByteTensor(box_target.size()).zero_().bool() # Create the mask
        contain_noobj_mask = torch.cuda.ByteTensor(box_target.size()).zero_().bool() 
        
        target_iou_gt = torch.zeros(box_target[:,0].size()).cuda() # In order to save the IoU for confidence 
                                                                   # of the cell which contains object
        
        for i in range(0, box_target.size()[0], 2):
            box_p = box_pred[i:i+2] # For 2 bounding boxes information
            box_p_coord = Variable(torch.FloatTensor(box_p.size())) # Box coordinate (x_min, y_min, x_max, y_max)
            # [2, 5=len([x, y, w, h, conf])]
            
            box_p_coord[:,:2] = box_p[:, :2]/float(7) - 0.5 * box_p[:, 2:4] ## Fill out 
            # box의 x_min과 y_min을 나타낸다. 여기서 7로 나누어준 이유는
            # data를 발생시킬 때 0,1사이로 nomalize해주었고 이를 다시 7등분했기 때문이다.
            box_p_coord[:,2:4] = box_p[:, :2]/float(7) + 0.5 * box_p[:, 2:4] ## Fill out
            
            box_t = box_target[i].contiguous().view(-1,5)
            box_t_coord = Variable(torch.FloatTensor(box_t.size()))
            box_t_coord[:,:2] = box_t[:, :2]/float(7) - 0.5 * box_t[:, 2:4]## Fill out
            box_t_coord[:,2:4] = box_t[:, :2]/float(7) + 0.5 * box_t[:, 2:4]## Fill out
            
            iou = self.compute_iou(box_p_coord[:,:4], box_t_coord[:,:4]) # [2,]
                                                      # compute IoU between prediction and target boxes, 
            
            gt_iou, max_idx = iou.max(0)
            contain_obj_mask[i+max_idx] = 1
            contain_noobj_mask[i+1-max_idx] = 1
            
            target_iou_gt[i+max_idx] = gt_iou.data.cuda()
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        target_iou_gt = target_iou_gt.to(device)
        
        ## For the cell which contains object
        contain_box_pred = box_pred[contain_obj_mask].view(-1, 5)  # [number_response, 5] ## Fill out 
        
        ## For the cell which does not contain object
        not_contain_box_pred = box_pred[contain_noobj_mask].view(-1, 5)## Fill out 

        ## For target domain 
        contain_box_target = box_target[contain_obj_mask].view(-1, 5) # [number_response, 5] Fill out
        not_contain_box_target = box_target[contain_noobj_mask].view(-1, 5)## Fill out
        not_contain_box_target[:,4] = 0
        
        ## For the GT IoU 
        contain_iou_target = target_iou_gt[contain_obj_mask].view(-1, 5) ## Fill out 
        
        obj_loss = F.mse_loss(contain_box_pred[:,4], contain_iou_target, size_average=False)
        xy_loss = F.mse_loss(contain_box_pred[:,:2], contain_box_target[:,:2], size_average=False)
        wh_loss = F.mse_loss(torch.sqrt(contain_box_pred[:,2:4]), torch.sqrt(contain_box_target[:,2:4]), size_average=False)
        noobj_loss2 = F.mse_loss(not_contain_box_pred[:,4], not_contain_box_target[:,4], size_average=False)
        
        return obj_loss, xy_loss, wh_loss, noobj_loss2
        
    def compute_iou(self, box_p, box_t):
        
        pred_area = (box_p[:,2] - box_p[:,0]) * (box_p[:,3] - box_p[:,1]) # [2,] 한 cell당 2개씩 예상하므로
        target_area = (box_t[:,2] - box_t[:,0]) * (box_t[:,3] - box_t[:,1]) #[1,] ground truth는 한개
        
        box_p = box_p.unsqueeze(1).expand(2,1,4)
        box_t = box_t.unsqueeze(0).expand(2,1,4)
        
        # pred과 target box가 있다고 생각해보자 그러면 교집합부분을 구하기 위해서는 작은 사각형의 꼭짓점의 좌표를 알아야한다.
        # 이때 pixel은 왼쪽 위부터 0,0으로 세기 때문에 lt는 교집합 사각형의 왼쪽 윗부분을 
        # rb는 교집합 사각형의 오른쪽 아랫부분을 지칭하게 된다. 
        
        inter_lt = torch.max(box_p[:,:,:2], box_t[:,:,:2]) # [2,1,2]
        
        inter_rb = torch.min(box_p[:,:,2:], box_t[:,:,2:]) # [2,1,2]
        
        wh = inter_rb - inter_lt # [2,1,2]
        wh[wh<0] = 0 # 음수라는 것은 교집합 부분이 없다는 뜻이다.
        
        inter_area = wh[:,:,0] * wh[:,:,1] # [2,1]
        pred_area = pred_area.unsqueeze(1).expand_as(inter_area)
        target_area = target_area.unsqueeze(0).expand_as(inter_area)
        
        iou = inter_area / (pred_area + target_area - inter_area)
        
        return iou # [2,1] -> 한 cell 당 2개의 bounding box의 iou가 출력으로 나오게 되는 것이다.
    
    
    def forward(self,pred,target):

        batch_size = pred.size()[0]
        
        noobj_loss1 = self.not_contain_obj_error(pred, target)
        prob_loss = self.compute_prob_error(pred, target)
        obj_loss, xy_loss, wh_loss, noobj_loss2 = self.contain_obj_error(pred, target)
        
        total_loss = (self.l_coord*(xy_loss + wh_loss) + obj_loss + self.l_noobj*(noobj_loss1 + noobj_loss2) + prob_loss)
        
        return total_loss / batch_size
        