import wandb

# WandB에 로그인
wandb.login()

# WandB Artifact 생성
artifact = wandb.Artifact(name='yolov5_hyp_evol', type='dataset')

# CSV 파일 경로 설정
csv_file_path = '/home/mmc-server4/Server/Users/hayeon/AUE8088/PA2/runs/train/yolov5_hyp_evol9/results.csv'

# CSV 파일 추가
with artifact.new_file('results.csv', mode='wb') as file:
    file.write(open(csv_file_path, 'rb').read())

# WandB에 Artifact 업로드
run = wandb.init(project='aue8088-pa2')
run.log_artifact(artifact)
