
import os
import time
import sys
import datetime
import torch
from utils import MIX, CAT, COMPUTE_LEN, LOOP
from api import tts, audio
wav_path = "/cpfs01/user/renyiming/AudiobookAgent/output_gradio/run_20250627_175932_1db552b77450449a8f00d90ac120c0aa/audio"
os.makedirs(wav_path, exist_ok=True)


def function_one():
    print("🚀 开始生成音效和背景音素材")
    start_time = time.time()
    audio(prompt="Reeds swaying in the wind, faint footsteps on broken wooden bridge", duration=6, volume=-24, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_0_Reeds_swaying_in_the_wind.wav"))
    audio(prompt="Footsteps of women in silk clothing approaching", duration=4, volume=-26, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_1_Footsteps_of_women_in_silk.wav"))
    audio(prompt="Footsteps echoing in a large ornate hall, distant flute playing", duration=6, volume=-26, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_2_Footsteps_echoing_in_a_large.wav"))
    audio(prompt="Footsteps fading into the distance over gravel", duration=6, volume=-26, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_3_Footsteps_fading_into_the_distance.wav"))
    audio(prompt="Sunset ambiance, warm wind, reeds rustling", volume=-38, duration=30, negative_prompt=" ", output_path=os.path.join(wav_path, "bg_music_0_Sunset_ambiance_warm_wind_reeds.wav"))
    torch.cuda.empty_cache()
    audio(prompt="Palace interior ambiance with faint traditional string music and incense", volume=-36, duration=30, negative_prompt=" ", output_path=os.path.join(wav_path, "bg_music_1_Palace_interior_ambiance_with_faint.wav"))
    torch.cuda.empty_cache()
    audio(prompt="Lonely wind blowing over desert terrain", volume=-37, duration=30, negative_prompt=" ", output_path=os.path.join(wav_path, "bg_music_2_Lonely_wind_blowing_over_desert.wav"))
    torch.cuda.empty_cache()
    end_time = time.time()
    print(f"🎉 音效和背景音素材生成完成，耗时 {end_time - start_time:.2f} 秒")

def function_two():
    print("🚀 开始生成配音")
    start_time = time.time()
    tts(tts_text="夕阳西下，余晖洒在长满芦苇的江岸边。师徒四人跨过一座断桥，缓缓踏入一片神秘的土地——女儿国。香风袭人，花影婆娑，一股异样的气息扑面而来。", prompt_text="整体恐怖事件，是从几个年轻人的一场无聊的游戏开始的。", prompt_speech_path="/cpfs01/user/renyiming/AudiobookAgent/TTS/CosyVoice/cvd/pb.wav", speaker="旁白", volume=-15, output_path=os.path.join(wav_path, "fg_speech_0_.wav"))
    tts(tts_text="师父，这地方……怎么处处是香气？我老孙的鼻子都快给熏晕了。", prompt_text="欸，师父，你忘了他们把你吊在树上的滋味了？哼，你要是心疼这帮毛贼，就让他们还把你吊在树上荡秋千，老孙再也不管了！", prompt_speech_path="/cpfs01/user/renyiming/AudiobookAgent/TTS/CosyVoice/cvd/swk.wav", speaker="孙悟空", volume=-15, output_path=os.path.join(wav_path, "fg_speech_1_.wav"))
    tts(tts_text="悟空，不可妄言。此地女子为尊，当礼数周全，不可造次。", prompt_text="阿弥陀佛，仙童一见那东西我就胆战心惊，哪里还敢偷着吃呀。", prompt_speech_path="/cpfs01/user/renyiming/AudiobookAgent/TTS/CosyVoice/cvd/ts.wav", speaker="唐僧", volume=-15, output_path=os.path.join(wav_path, "fg_speech_2_.wav"))
    tts(tts_text="嘿嘿，师父说得对。不过这女儿国啊，果然如传说中那般，个个貌美如花，俺也得收敛点，收敛点……", prompt_text="你在这看着师父，我去把白龙马牵到有人家的地方买了，咱们把师父埋了呢，咱们只好是各奔东西啦。", prompt_speech_path="/cpfs01/user/renyiming/AudiobookAgent/TTS/CosyVoice/cvd/zbj.wav", speaker="猪八戒", volume=-15, output_path=os.path.join(wav_path, "fg_speech_3_.wav"))
    tts(tts_text="师父，前方好像有宫殿守卫，我等还是先停步，听候师命。", prompt_text="阿弥陀佛，仙童一见那东西我就胆战心惊，哪里还敢偷着吃呀。", prompt_speech_path="/cpfs01/user/renyiming/AudiobookAgent/TTS/CosyVoice/cvd/ts.wav", speaker="沙僧", volume=-15, output_path=os.path.join(wav_path, "fg_speech_4_.wav"))
    tts(tts_text="正当四人打量四周之时，几名着锦衣华服的宫女迎面而来，个个仪态端庄，目光含笑，将他们团团围住。", prompt_text="整体恐怖事件，是从几个年轻人的一场无聊的游戏开始的。", prompt_speech_path="/cpfs01/user/renyiming/AudiobookAgent/TTS/CosyVoice/cvd/pb.wav", speaker="旁白", volume=-15, output_path=os.path.join(wav_path, "fg_speech_5_.wav"))
    tts(tts_text="几位远方高人，乃何方而来？为何擅入我女儿国境？", prompt_text="你宣称自己为和平而来，但我们对和平的理解似乎并不相同。", prompt_speech_path="/cpfs01/user/renyiming/AudiobookAgent/TTS/CosyVoice/cvd/znw.wav", speaker="宫女", volume=-15, output_path=os.path.join(wav_path, "fg_speech_6_.wav"))
    tts(tts_text="贫僧自东土大唐而来，前往西天拜佛取经，途经贵境，望恕叨扰。", prompt_text="阿弥陀佛，仙童一见那东西我就胆战心惊，哪里还敢偷着吃呀。", prompt_speech_path="/cpfs01/user/renyiming/AudiobookAgent/TTS/CosyVoice/cvd/ts.wav", speaker="唐僧", volume=-15, output_path=os.path.join(wav_path, "fg_speech_7_.wav"))
    tts(tts_text="原是取经圣僧。国主听闻贵人驾到，特命奴婢前来相迎。请随我等入宫稍作歇息。", prompt_text="你宣称自己为和平而来，但我们对和平的理解似乎并不相同。", prompt_speech_path="/cpfs01/user/renyiming/AudiobookAgent/TTS/CosyVoice/cvd/znw.wav", speaker="宫女", volume=-15, output_path=os.path.join(wav_path, "fg_speech_8_.wav"))
    tts(tts_text="宫女引领师徒进入女儿国王宫，雕梁画栋，香雾缭绕。女儿国国王端坐于高台之上，一袭凤袍，眉眼如画，目光在唐僧身上一顿，眉头轻蹙，心潮起伏。", prompt_text="整体恐怖事件，是从几个年轻人的一场无聊的游戏开始的。", prompt_speech_path="/cpfs01/user/renyiming/AudiobookAgent/TTS/CosyVoice/cvd/pb.wav", speaker="旁白", volume=-15, output_path=os.path.join(wav_path, "fg_speech_9_.wav"))
    tts(tts_text="这位法师，风度翩翩，仪表不凡。常言道：大千世界，情为何物？本王愿与法师一叙情缘，可否？", prompt_text="你宣称自己为和平而来，但我们对和平的理解似乎并不相同。", prompt_speech_path="/cpfs01/user/renyiming/AudiobookAgent/TTS/CosyVoice/cvd/znw.wav", speaker="女儿国国王", volume=-15, output_path=os.path.join(wav_path, "fg_speech_10_.wav"))
    tts(tts_text="贫僧一心向佛，不敢染尘缘。国主美意，唐某心领，但还望勿执。", prompt_text="阿弥陀佛，仙童一见那东西我就胆战心惊，哪里还敢偷着吃呀。", prompt_speech_path="/cpfs01/user/renyiming/AudiobookAgent/TTS/CosyVoice/cvd/ts.wav", speaker="唐僧", volume=-15, output_path=os.path.join(wav_path, "fg_speech_11_.wav"))
    tts(tts_text="哎呀，师父，这国王对您真是一片深情，您可得斟酌啊，咱取经的路也不能太苦不是？", prompt_text="你在这看着师父，我去把白龙马牵到有人家的地方买了，咱们把师父埋了呢，咱们只好是各奔东西啦。", prompt_speech_path="/cpfs01/user/renyiming/AudiobookAgent/TTS/CosyVoice/cvd/zbj.wav", speaker="猪八戒", volume=-15, output_path=os.path.join(wav_path, "fg_speech_12_.wav"))
    tts(tts_text="呆子！你又胡说八道，师父定力如山，怎会动凡心？", prompt_text="欸，师父，你忘了他们把你吊在树上的滋味了？哼，你要是心疼这帮毛贼，就让他们还把你吊在树上荡秋千，老孙再也不管了！", prompt_speech_path="/cpfs01/user/renyiming/AudiobookAgent/TTS/CosyVoice/cvd/swk.wav", speaker="孙悟空", volume=-15, output_path=os.path.join(wav_path, "fg_speech_13_.wav"))
    tts(tts_text="国王一片真情，但我师徒肩负天命，实不宜久留，还望见谅。", prompt_text="阿弥陀佛，仙童一见那东西我就胆战心惊，哪里还敢偷着吃呀。", prompt_speech_path="/cpfs01/user/renyiming/AudiobookAgent/TTS/CosyVoice/cvd/ts.wav", speaker="沙僧", volume=-15, output_path=os.path.join(wav_path, "fg_speech_14_.wav"))
    tts(tts_text="女儿国王神色复杂，望着唐僧渐行渐远的背影，泪光盈盈。师徒四人离开王宫，踏上取经之路，风沙中，情意如幻影消散在远方。", prompt_text="整体恐怖事件，是从几个年轻人的一场无聊的游戏开始的。", prompt_speech_path="/cpfs01/user/renyiming/AudiobookAgent/TTS/CosyVoice/cvd/pb.wav", speaker="旁白", volume=-15, output_path=os.path.join(wav_path, "fg_speech_15_.wav"))
    end_time = time.time()
    print(f"🎉 配音生成完成，耗时 {end_time - start_time:.2f} 秒")
print("🚀 开始生成音频文件")
start_time = time.time()
p1 = Process(target=function_one)
p2 = Process(target=function_two)
p1.start()
p2.start()
p1.join()
p2.join()
fg_audio_wavs = []
fg_audio_lens = []
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_0_.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_0_.wav")))

fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_0_Reeds_swaying_in_the_wind.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_0_Reeds_swaying_in_the_wind.wav")))

fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_1_.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_1_.wav")))

fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_2_.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_2_.wav")))

fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_3_.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_3_.wav")))

fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_4_.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_4_.wav")))

fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_1_Footsteps_of_women_in_silk.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_1_Footsteps_of_women_in_silk.wav")))

fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_5_.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_5_.wav")))

fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_6_.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_6_.wav")))

fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_7_.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_7_.wav")))

fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_8_.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_8_.wav")))

fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_2_Footsteps_echoing_in_a_large.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_2_Footsteps_echoing_in_a_large.wav")))

fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_9_.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_9_.wav")))

fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_10_.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_10_.wav")))

fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_11_.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_11_.wav")))

fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_12_.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_12_.wav")))

fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_13_.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_13_.wav")))

fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_14_.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_14_.wav")))

fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_3_Footsteps_fading_into_the_distance.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_3_Footsteps_fading_into_the_distance.wav")))

fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_15_.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_15_.wav")))

CAT(wavs=fg_audio_wavs, out_wav=os.path.join(wav_path, "foreground.wav"))
print("🚀 开始处理背景音并生成混音")

bg_audio_offsets = []
bg_audio_len = sum(fg_audio_lens[0:11])
bg_audio_offset = sum(fg_audio_lens[:0])
LOOP(os.path.join(wav_path, "bg_music_0_Sunset_ambiance_warm_wind_reeds.wav"), os.path.join(wav_path, "bg_music_0_Sunset_ambiance_warm_wind_reeds.wav"), bg_audio_len)
bg_audio_offsets.append(bg_audio_offset)

bg_audio_len = sum(fg_audio_lens[11:18])
bg_audio_offset = sum(fg_audio_lens[:11])
LOOP(os.path.join(wav_path, "bg_music_1_Palace_interior_ambiance_with_faint.wav"), os.path.join(wav_path, "bg_music_1_Palace_interior_ambiance_with_faint.wav"), bg_audio_len)
bg_audio_offsets.append(bg_audio_offset)

bg_audio_len = sum(fg_audio_lens[18:20])
bg_audio_offset = sum(fg_audio_lens[:18])
LOOP(os.path.join(wav_path, "bg_music_2_Lonely_wind_blowing_over_desert.wav"), os.path.join(wav_path, "bg_music_2_Lonely_wind_blowing_over_desert.wav"), bg_audio_len)
bg_audio_offsets.append(bg_audio_offset)

bg_audio_wavs = []
bg_audio_lens = []
bg_audio_wavs.append(os.path.join(wav_path, "bg_music_0_Sunset_ambiance_warm_wind_reeds.wav"))
bg_audio_wavs.append(os.path.join(wav_path, "bg_music_1_Palace_interior_ambiance_with_faint.wav"))
bg_audio_wavs.append(os.path.join(wav_path, "bg_music_2_Lonely_wind_blowing_over_desert.wav"))
bg_audio_wav_offset_pairs = list(zip(bg_audio_wavs, bg_audio_offsets))
bg_audio_wav_offset_pairs.append((os.path.join(wav_path, "foreground.wav"), 0))
MIX(wavs=bg_audio_wav_offset_pairs, out_wav=os.path.join(wav_path, "final_mix.wav"))
end_time = time.time()
print(f"🎉 音频生成完成，耗时 {end_time - start_time:.2f} 秒")
