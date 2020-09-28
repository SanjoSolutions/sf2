class VideoRecorder:
  __init__(self, emulator, monitor_csv=None, video_file=None, info_file=None, npy_file=None, viewer=None, video_delay=0, lossless=None, record_audio=True):
    ffmpeg_proc = None
    viewer_proc = None
    info_steps = []
    actions = np.empty(shape=(0, emulator.num_buttons * emulator.players), dtype=bool)
    if viewer or video_file:
      video = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      video.bind(('127.0.0.1', 0))
      vr = video.getsockname()[1]
      input_vformat = [
        '-r', str(emulator.em.get_screen_rate()),
        '-s', '%dx%d' % emulator.observation_space.shape[1::-1],
        '-pix_fmt', 'rgb24',
        '-f', 'rawvideo',
        '-probesize', '32',
        '-thread_queue_size', '10000',
        '-i', 'tcp://127.0.0.1:%i?listen' % vr
      ]
      if record_audio:
        audio = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        audio.bind(('127.0.0.1', 0))
        ar = audio.getsockname()[1]
        input_aformat = [
          '-ar', '%i' % emulator.em.get_audio_rate(),
          '-ac', '2',
          '-f', 's16le',
          '-probesize', '32',
          '-thread_queue_size', '60',
          '-i', 'tcp://127.0.0.1:%i?listen' % ar
        ]
      else:
        audio = None
        ar = None
        input_aformat = ['-an']
      stdout = None
      output = []
      if video_file:
        if not lossless:
          output = ['-c:a', 'aac', '-b:a', '128k', '-strict', '-2', '-c:v', 'libx264', '-preset', 'slow', '-crf', '17', '-f', 'mp4', '-pix_fmt', 'yuv420p', video_file]
        elif lossless == 'mp4':
          output = ['-c:a', 'aac', '-b:a', '192k', '-strict', '-2', '-c:v', 'libx264', '-preset', 'veryslow', '-crf', '0', '-f', 'mp4', '-pix_fmt', 'yuv444p', video_file]
        elif lossless == 'mp4rgb':
          output = ['-c:a', 'aac', '-b:a', '192k', '-strict', '-2', '-c:v', 'libx264rgb', '-preset', 'veryslow', '-crf', '0', '-f', 'mp4', '-pix_fmt', 'rgb24', video_file]
        elif lossless == 'png':
          output = ['-c:a', 'flac', '-c:v', 'png', '-pix_fmt', 'rgb24', '-f', 'matroska', video_file]
        elif lossless == 'ffv1':
          output = ['-c:a', 'flac', '-c:v', 'ffv1', '-pix_fmt', 'bgr0', '-f', 'matroska', video_file]
      if viewer:
        stdout = subprocess.PIPE
        output = ['-c', 'copy', '-f', 'nut', 'pipe:1']
      ffmpeg_proc = subprocess.Popen(['ffmpeg', '-y',
        *input_vformat,  # Input params (video)
        *input_aformat,  # Input params (audio)
        *output],  # Output params
        stdout=stdout)
      video.close()
      video = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      if audio:
        audio.close()
        audio = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      audio_connected = False

      time.sleep(0.3)
      try:
        video.connect(('127.0.0.1', vr))
      except ConnectionRefusedError:
        video.close()
        if audio:
            audio.close()
        ffmpeg_proc.terminate()
        raise
      if viewer:
        viewer_proc = subprocess.Popen([viewer, '-'], stdin=ffmpeg_proc.stdout)
    frames = 0
    score = [0] * emulator.players
    reward_fields = ['r'] if emulator.players == 1 else ['r%d' % i for i in range(emulator.players)]
    wasDone = False

  record_frame(self):
    if info_file:
      info_steps.append(info)
    if emulator.players > 1:
      for p in range(emulator.players):
        score[p] += reward[p]
    else:
      score[0] += reward
    frames += 1
    try:
      if hasattr(signal, 'SIGCHLD'):
        signal.signal(signal.SIGCHLD, self._killprocs)
      if viewer_proc and viewer_proc.poll() is not None:
        break
      if ffmpeg_proc and frames > video_delay:
        video.sendall(bytes(display))
        if audio:
          sound = emulator.em.get_audio()
          if not audio_connected:
            time.sleep(0.2)
            audio.connect(('127.0.0.1', ar))
            audio_connected = True
          if len(sound):
            audio.sendall(bytes(sound))
    except BrokenPipeError:
      self._waitprocs()
      raise
    finally:
      if hasattr(signal, 'SIGCHLD'):
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    if done and not wasDone:
      if monitor_csv:
          monitor_csv.writerow({**dict(zip(reward_fields, score)), 'l': frames, 't': frames / 60.0})
      frames = 0
      score = [0] * emulator.players
    wasDone = done
    if hasattr(signal, 'SIGCHLD'):
      signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    if monitor_csv and frames:
      monitor_csv.writerow({**dict(zip(reward_fields, score)), 'l': frames, 't': frames / 60.0})
    if npy_file:
      kwargs = {
        'actions': actions
      }
      if info_file:
        kwargs['info'] = info_steps
      try:
        np.savez_compressed(npy_file, **kwargs)
      except IOError:
        pass
    elif info_file:
      try:
        with open(info_file, 'w') as f:
          json.dump(info_steps, f)
      except IOError:
        pass

    stop():
      self._waitprocs()

    def _killprocs(*args, **kwargs):
      ffmpeg_proc.terminate()
      if viewer:
          viewer_proc.terminate()
          viewer_proc.wait()
      raise BrokenPipeError

    def _waitprocs():
      if ffmpeg_proc:
        video.close()
        if audio:
          audio.close()
        if not viewer_proc or viewer_proc.poll() is None:
          ffmpeg_proc.wait()
