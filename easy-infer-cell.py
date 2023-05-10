%cd /content/Retrieval-based-Voice-Conversion-WebUI
from multiprocessing import cpu_count
import threading, pdb, librosa
from time import sleep
from subprocess import Popen
from time import sleep
import torch, os, traceback, sys, warnings, shutil, numpy as np
import faiss
from random import shuffle
import scipy.io.wavfile as wavfile
from mega import Mega

now_dir = os.getcwd()
sys.path.append(now_dir)
tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs("audios",exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "weights"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)
from i18n import I18nAuto
import ffmpeg
import datetime
import subprocess

print("Epic")

i18n = I18nAuto()
# 判断是否有能用来训练和加速推理的N卡
ncpu = cpu_count()
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if (not torch.cuda.is_available()) or ngpu == 0:
    if_gpu_ok = False
else:
    if_gpu_ok = False
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if (
            "10" in gpu_name
            or "16" in gpu_name
            or "20" in gpu_name
            or "30" in gpu_name
            or "40" in gpu_name
            or "A2" in gpu_name.upper()
            or "A3" in gpu_name.upper()
            or "A4" in gpu_name.upper()
            or "P4" in gpu_name.upper()
            or "A50" in gpu_name.upper()
            or "70" in gpu_name
            or "80" in gpu_name
            or "90" in gpu_name
            or "M4" in gpu_name.upper()
            or "T4" in gpu_name.upper()
            or "TITAN" in gpu_name.upper()
        ):  # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(
                int(
                    torch.cuda.get_device_properties(i).total_memory
                    / 1024
                    / 1024
                    / 1024
                    + 0.4
                )
            )

print("GPU CHeck")
if if_gpu_ok == True and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = "很遗憾您这没有能用的显卡来支持您训练"
    default_batch_size = 1
gpus = "-".join([i[0] for i in gpu_infos])

print("Time to import")
from infer_pack.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono
from scipy.io import wavfile
from fairseq import checkpoint_utils
import gradio as gr
import logging
from vc_infer_pipeline import VC
from config import (
    is_half,
    device,
    #python_cmd,
    #listen_port,
    #iscolab,
    #noparallel,
    #noautoopen,
)
# FIX
iscolab = True
noautoopen = False

from infer_uvr5 import _audio_pre_
from my_utils import load_audio
from train.process_ckpt import show_info, change_info, merge, extract_small_model

print("Checking log")
# from trainset_preprocess_pipeline import PreProcess
logging.getLogger("numba").setLevel(logging.WARNING)

print("Epic2")


class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"


hubert_model = None


def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(device)
    if is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()


weight_root = "weights"
weight_uvr5_root = "uvr5_weights"
names = []
for name in os.listdir(weight_root):
    if name.endswith(".pth"):
        names.append(name)
        
uvr5_names = []
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth"):
        uvr5_names.append(name.replace(".pth", ""))

def find_parent(search_dir, file_name):
    for dirpath, dirnames, filenames in os.walk(search_dir):
        if file_name in filenames:
            return os.path.abspath(dirpath)
    return None

def vc_single(
    sid,
    input_audio,
    f0_up_key,
    f0_file,
    f0_method,
    file_index,
    # file_big_npy,
    index_rate,
):  # spk_item, input_audio0, vc_transform0,f0_file,f0method0
    global tgt_sr, net_g, vc, hubert_model
    if input_audio is None:
        return "You need to upload an audio", None
    f0_up_key = int(f0_up_key)
    try:
        parent_dir = find_parent(".",input_audio)
        audio = load_audio(parent_dir+'/'+input_audio, 16000)
        times = [0, 0, 0]
        if hubert_model == None:
            load_hubert()
        if_f0 = cpt.get("f0", 1)
        try:
            file_index = (
                file_index.strip(" ")
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip(" ")
                .replace("trained", "added")
            )  # 防止小白写错，自动帮他替换掉
        except:
            file_index=''
            print("Skipped index.")
        # file_big_npy = (
        #     file_big_npy.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        # )
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            sid,
            audio,
            times,
            f0_up_key,
            f0_method,
            file_index,
            # file_big_npy,
            index_rate,
            if_f0,
            f0_file=f0_file,
        )
        print(
            "npy: ", times[0], "s, f0: ", times[1], "s, infer: ", times[2], "s", sep=""
        )
        return "Success", (tgt_sr, audio_opt)
    except:
        info = traceback.format_exc()
        print(info)
        return info, (None, None)


def vc_multi(
    sid,
    dir_path,
    opt_root,
    paths,
    f0_up_key,
    f0_method,
    file_index,
    # file_big_npy,
    index_rate,
):
    try:
        dir_path = (
            dir_path.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )  # 防止小白拷路径头尾带了空格和"和回车
        opt_root = opt_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        os.makedirs(opt_root, exist_ok=True)
        try:
            if dir_path != "":
                paths = [os.path.join(dir_path, name) for name in os.listdir(dir_path)]
            else:
                paths = [path.name for path in paths]
        except:
            traceback.print_exc()
            paths = [path.name for path in paths]
        infos = []
        file_index = (
            file_index.strip(" ")
            .strip('"')
            .strip("\n")
            .strip('"')
            .strip(" ")
            .replace("trained", "added")
        )  # 防止小白写错，自动帮他替换掉
        for path in paths:
            info, opt = vc_single(
                sid,
                path,
                f0_up_key,
                None,
                f0_method,
                file_index,
                # file_big_npy,
                index_rate,
            )
            if info == "Success":
                try:
                    tgt_sr, audio_opt = opt
                    wavfile.write(
                        "%s/%s" % (opt_root, os.path.basename(path)), tgt_sr, audio_opt
                    )
                except:
                    info = traceback.format_exc()
            infos.append("%s->%s" % (os.path.basename(path), info))
            yield "\n".join(infos)
        yield "\n".join(infos)
    except:
        yield traceback.format_exc()

# 一个选项卡全局只能有一个音色
def get_vc(sid):
    global n_spk, tgt_sr, net_g, vc, cpt
    if sid == []:
        global hubert_model
        if hubert_model != None:  # 考虑到轮询, 需要加个判断看是否 sid 是由有模型切换到无模型的
            print("clean_empty_cache")
            del net_g, n_spk, vc, hubert_model, tgt_sr  # ,cpt
            hubert_model = net_g = n_spk = vc = hubert_model = tgt_sr = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            ###楼下不这么折腾清理不干净
            if_f0 = cpt.get("f0", 1)
            if if_f0 == 1:
                net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
            else:
                net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
            del net_g, cpt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cpt = None
        return {"visible": False, "__type__": "update"}
    person = "%s/%s" % (weight_root, sid)
    print("loading %s" % person)
    cpt = torch.load(person, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    if if_f0 == 1:
        net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
    else:
        net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))  # 不加这一行清不干净, 真奇葩
    net_g.eval().to(device)
    if is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, device, is_half)
    n_spk = cpt["config"][-3]
    return {"visible": False, "maximum": n_spk, "__type__": "update"}


def change_choices():
    names = []
    for name in os.listdir(weight_root):
        if name.endswith(".pth"):
            names.append(name)
    return {"choices": sorted(names), "__type__": "update"}

def change_choices2():
    audio_files=[]
    for filename in os.listdir("./audios"):
        if filename.endswith(('.wav','.mp3')):
            audio_files.append(filename)
    return {"choices": sorted(audio_files), "__type__": "update"}

def clean():
    return {"value": "", "__type__": "update"}

def change_sr2(sr2, if_f0_3):
    if if_f0_3 == "是":
        return "pretrained/f0G%s.pth" % sr2, "pretrained/f0D%s.pth" % sr2
    else:
        return "pretrained/G%s.pth" % sr2, "pretrained/D%s.pth" % sr2
        
def get_index():
    if check_for_name() != '':
        if iscolab:
            chosen_model=sorted(names)[0].split(".")[0]
            logs_path="/content/Retrieval-based-Voice-Conversion-WebUI/logs/"+chosen_model
            for file in os.listdir(logs_path):
                if file.endswith(".index"):
                    return os.path.join(logs_path, file)
            return ''
        else:
            return ''
        
def get_indexes():
    indexes_list=[]
    if iscolab:
        for dirpath, dirnames, filenames in os.walk("/content/Retrieval-based-Voice-Conversion-WebUI/logs/"):
            for filename in filenames:
                if filename.endswith(".index"):
                    indexes_list.append(os.path.join(dirpath,filename))
        return indexes_list
    else:
        return ''
        
audio_files=[]
for filename in os.listdir("./audios"):
    if filename.endswith(('.wav','.mp3')):
        audio_files.append(filename)
        
def get_name():
    if len(audio_files) > 0:
        return sorted(audio_files)[0]
    else:
        return ''
        
def save_to_wav(record_button):
    if record_button is None:
        pass
    else:
        path_to_file=record_button
        new_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+'.wav'
        new_path='./audios/'+new_name
        shutil.move(path_to_file,new_path)
        return new_name
    
def save_to_wav2(dropbox):
    file_path=dropbox.name
    shutil.move(file_path,'./audios')
    return os.path.basename(file_path)
    
def match_index(speaker):
    folder=speaker.split(".")[0]
    parent_dir="/content/Retrieval-based-Voice-Conversion-WebUI/logs/"+folder
    for filename in os.listdir(parent_dir):
        if filename.endswith(".index"):
            index_path=os.path.join(parent_dir,filename)
            return index_path
            
def download_from_url(url, model):
    url = url.strip()
    if url == '':
        return "URL cannot be left empty."
    zip_dirs = ["zips", "unzips"]
    for directory in zip_dirs:
        if os.path.exists(directory):
            shutil.rmtree(directory)
    os.makedirs("zips", exist_ok=True)
    os.makedirs("unzips", exist_ok=True)
    zipfile = model + '.zip'
    zipfile_path = './zips/' + zipfile
    MODELEPOCH = ''
    if "drive.google.com" in url:
        subprocess.run(["gdown", url, "--fuzzy", "-O", zipfile_path])
    elif "mega.nz" in url:
        m = Mega()
        m.download_url(url, './zips')
    else:
        subprocess.run(["wget", url, "-O", f"./zips/{zipfile}"])
    for filename in os.listdir("./zips"):
        if filename.endswith(".zip"):
            zipfile_path = os.path.join("./zips/",filename)
            shutil.unpack_archive(zipfile_path, "./unzips", 'zip')
        else:
            return "No zipfile found."
    for root, dirs, files in os.walk('./unzips'):
        for file in files:
            if "G_" in file:
                MODELEPOCH = file.split("G_")[1].split(".")[0]
        if MODELEPOCH == '':
            MODELEPOCH = '404'
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".npy") or file.endswith(".index"):
                subprocess.run(["mkdir", "-p", f"./logs/{model}"])
                subprocess.run(["mv", file_path, f"./logs/{model}/"])
            elif "G_" not in file and "D_" not in file and file.endswith(".pth"):
                subprocess.run(["mv", file_path, f"./weights/{model}.pth"])
    shutil.rmtree("zips")
    shutil.rmtree("unzips")
    return "Success."

def check_for_name():
    if len(names) > 0:
        return sorted(names)[0]
    else:
        return ''
print(check_for_name())        
#with gr.Blocks() as app
with gr.Blocks(theme=gr.themes.Base()) as app:
    with gr.Tab("Inference"):
        with gr.Row():
            sid0 = gr.Dropdown(label="1.Choose your Model.", choices=sorted(names), value=check_for_name())
            refresh_button = gr.Button("Refresh", variant="primary", size='sm')
            refresh_button.click(fn=change_choices, inputs=[], outputs=[sid0])
            if check_for_name() != '':
                get_vc(sorted(names)[0])
            else:
                print("Starting without preloaded Model.")
            vc_transform0 = gr.Number(label="Optional: You can change the pitch here or leave it at 0.", value=0)
            #clean_button = gr.Button("Unload Voice to Save Memory", variant="primary")
            spk_item = gr.Slider(minimum=0,maximum=2333,step=1,label="Please select speaker id",value=0,visible=False,interactive=True)
            #clean_button.click(fn=clean, inputs=[], outputs=[sid0])
            sid0.change(
                fn=get_vc,
                inputs=[sid0],
                outputs=[],
            )
            but0 = gr.Button("Convert", variant="primary")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    dropbox = gr.File(label="Drop your audio here & hit the Reload button.")
                with gr.Row():
                    record_button=gr.Audio(source="microphone", label="OR Record audio.", type="filepath")
                with gr.Row():
                #input_audio0 = gr.Textbox(label="Enter the Path to the Audio File to be Processed (e.g. /content/youraudio.wav)",value="/content/youraudio.wav")
                    input_audio0 = gr.Dropdown(choices=sorted(audio_files), label="2.Choose your audio.", value=get_name())
                    dropbox.upload(fn=save_to_wav2, inputs=[dropbox], outputs=[input_audio0])
                    dropbox.upload(fn=change_choices2, inputs=[], outputs=[input_audio0])
                    refresh_button2 = gr.Button("Refresh", variant="primary", size='sm')
                    refresh_button2.click(fn=change_choices2, inputs=[], outputs=[input_audio0])
                    record_button.change(fn=save_to_wav, inputs=[record_button], outputs=[input_audio0])
                    record_button.change(fn=change_choices2, inputs=[], outputs=[input_audio0])
            with gr.Column():
                with gr.Accordion(label="Feature Settings", open=False):
                    file_index1 = gr.Dropdown(
                        label="3. Path to your added.index file (if it didn't automatically find it.)",
                        value=get_index(),
                        choices=get_indexes(),
                        interactive=True,
                        visible=True
                    )
                    index_rate1 = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Strength:",
                    value=0.69,
                    interactive=True,
                    )
                sid0.change(fn=match_index, inputs=[sid0], outputs=[file_index1])
                
                with gr.Row():
                    vc_output2 = gr.Audio(label="Output Audio (Click on the Three Dots in the Right Corner to Download)") 
                with gr.Row():
                    f0method0 = gr.Radio(
                    label="Optional: Change the Pitch Extraction Algorithm. Use PM for fast results or Harvest for better low range (but it's extremely slow)",
                    choices=["pm", "harvest"],
                    value="pm",
                    interactive=True,
                    )
        with gr.Row():
            vc_output1 = gr.Textbox(label="")
        with gr.Row():
            instructions = gr.Markdown("""
                This is simply a modified version of the RVC GUI found here: 
                https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
                """)
        f0_file = gr.File(label="F0 Curve File (Optional, One Pitch Per Line, Replaces Default F0 and Pitch Shift)", visible=False)        
        but0.click(
            vc_single,
            [
                spk_item,
                input_audio0,
                vc_transform0,
                f0_file,
                f0method0,
                file_index1,
                index_rate1,
            ],
            [vc_output1, vc_output2]
        )
    with gr.Tab("Download Model"):
        with gr.Row():
            url=gr.Textbox(label="Enter the URL to the Model:")
        with gr.Row():
            model = gr.Textbox(label="Name your model:")
            download_button=gr.Button(label="Download")
        with gr.Row():
            status_bar=gr.Textbox(label="")
            download_button.click(fn=download_from_url, inputs=[url, model], outputs=[status_bar])
    if iscolab:
        app.launch(debug=True)
    else:
        app.queue(concurrency_count=511, max_size=1022).launch(
            server_name="0.0.0.0",
            inbrowser=not noautoopen,
            server_port=listen_port,
            quiet=True,
        )
