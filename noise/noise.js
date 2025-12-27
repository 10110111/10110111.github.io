// FFT implementation from https://github.com/dntj/jsfft v0.0.4
class baseComplexArray {
  constructor(other, arrayType = Float32Array) {
    if (other instanceof baseComplexArray) {
      // Copy constuctor.
      this.ArrayType = other.ArrayType;
      this.real = new this.ArrayType(other.real);
      this.imag = new this.ArrayType(other.imag);
    } else {
      this.ArrayType = arrayType;
      // other can be either an array or a number.
      this.real = new this.ArrayType(other);
      this.imag = new this.ArrayType(this.real.length);
    }

    this.length = this.real.length;
  }

  forEach(iterator) {
    const n = this.length;
    // For gc efficiency, re-use a single object in the iterator.
    const value = Object.seal(Object.defineProperties({}, {
      real: {writable: true}, imag: {writable: true},
    }));

    for(let i = 0; i < n; i++) {
      value.real = this.real[i];
      value.imag = this.imag[i];
      iterator(value, i, n);
    }
  }

  // In-place mapper.
  map(mapper) {
    this.forEach((value, i, n) => {
      mapper(value, i, n);
      this.real[i] = value.real;
      this.imag[i] = value.imag;
    });

    return this;
  }
}

// Math constants and functions we need.
const PI = Math.PI;
const SQRT1_2 = Math.SQRT1_2;

function FFT(input) {
  return ensureComplexArray(input).FFT();
};

function InvFFT(input) {
  return ensureComplexArray(input).InvFFT();
};

class ComplexArray extends baseComplexArray {
  FFT() {
    return fft(this, false);
  }

  InvFFT() {
    return fft(this, true);
  }
}

function ensureComplexArray(input) {
  return input instanceof ComplexArray && input || new ComplexArray(input);
}

function fft(input, inverse) {
  const n = input.length;

  if (n & (n - 1)) {
    return FFT_Recursive(input, inverse);
  } else {
    return FFT_2_Iterative(input, inverse);
  }
}

function FFT_Recursive(input, inverse) {
  const n = input.length;

  if (n === 1) {
    return input;
  }

  const output = new ComplexArray(n, input.ArrayType);

  // Use the lowest odd factor, so we are able to use FFT_2_Iterative in the
  // recursive transforms optimally.
  const p = LowestOddFactor(n);
  const m = n / p;
  const normalisation = 1 / Math.sqrt(p);
  let recursive_result = new ComplexArray(m, input.ArrayType);

  // Loops go like O(n Σ p_i), where p_i are the prime factors of n.
  // for a power of a prime, p, this reduces to O(n p log_p n)
  for(let j = 0; j < p; j++) {
    for(let i = 0; i < m; i++) {
      recursive_result.real[i] = input.real[i * p + j];
      recursive_result.imag[i] = input.imag[i * p + j];
    }
    // Don't go deeper unless necessary to save allocs.
    if (m > 1) {
      recursive_result = fft(recursive_result, inverse);
    }

    const del_f_r = Math.cos(2*PI*j/n);
    const del_f_i = (inverse ? -1 : 1) * Math.sin(2*PI*j/n);
    let f_r = 1;
    let f_i = 0;

    for(let i = 0; i < n; i++) {
      const _real = recursive_result.real[i % m];
      const _imag = recursive_result.imag[i % m];

      output.real[i] += f_r * _real - f_i * _imag;
      output.imag[i] += f_r * _imag + f_i * _real;

      [f_r, f_i] = [
        f_r * del_f_r - f_i * del_f_i,
        f_i = f_r * del_f_i + f_i * del_f_r,
      ];
    }
  }

  // Copy back to input to match FFT_2_Iterative in-placeness
  // TODO: faster way of making this in-place?
  for(let i = 0; i < n; i++) {
    input.real[i] = normalisation * output.real[i];
    input.imag[i] = normalisation * output.imag[i];
  }

  return input;
}

function FFT_2_Iterative(input, inverse) {
  const n = input.length;

  const output = BitReverseComplexArray(input);
  const output_r = output.real;
  const output_i = output.imag;
  // Loops go like O(n log n):
  //   width ~ log n; i,j ~ n
  let width = 1;
  while (width < n) {
    const del_f_r = Math.cos(PI/width);
    const del_f_i = (inverse ? -1 : 1) * Math.sin(PI/width);
    for (let i = 0; i < n/(2*width); i++) {
      let f_r = 1;
      let f_i = 0;
      for (let j = 0; j < width; j++) {
        const l_index = 2*i*width + j;
        const r_index = l_index + width;

        const left_r = output_r[l_index];
        const left_i = output_i[l_index];
        const right_r = f_r * output_r[r_index] - f_i * output_i[r_index];
        const right_i = f_i * output_r[r_index] + f_r * output_i[r_index];

        output_r[l_index] = SQRT1_2 * (left_r + right_r);
        output_i[l_index] = SQRT1_2 * (left_i + right_i);
        output_r[r_index] = SQRT1_2 * (left_r - right_r);
        output_i[r_index] = SQRT1_2 * (left_i - right_i);

        [f_r, f_i] = [
          f_r * del_f_r - f_i * del_f_i,
          f_r * del_f_i + f_i * del_f_r,
        ];
      }
    }
    width <<= 1;
  }

  return output;
}

function BitReverseIndex(index, n) {
  let bitreversed_index = 0;

  while (n > 1) {
    bitreversed_index <<= 1;
    bitreversed_index += index & 1;
    index >>= 1;
    n >>= 1;
  }
  return bitreversed_index;
}

function BitReverseComplexArray(array) {
  const n = array.length;
  const flips = new Set();

  for(let i = 0; i < n; i++) {
    const r_i = BitReverseIndex(i, n);

    if (flips.has(i)) continue;

    [array.real[i], array.real[r_i]] = [array.real[r_i], array.real[i]];
    [array.imag[i], array.imag[r_i]] = [array.imag[r_i], array.imag[i]];

    flips.add(r_i);
  }

  return array;
}

function LowestOddFactor(n) {
  const sqrt_n = Math.sqrt(n);
  let factor = 3;

  while(factor <= sqrt_n) {
    if (n % factor === 0) return factor;
    factor += 2;
  }
  return n;
}

// End of the FFT implementation

// Actual implementation of the noise generator
window.onload = function() {
  const startNoiseBtn = document.querySelector("#startNoiseBtn");
  const stopNoiseBtn = document.querySelector("#stopNoiseBtn");

  const frameCount = 2**16;

  const updatePeriod = 2.0; // seconds
  const numChannels = 2;
  const totalSources = 2;
  let audioCtx, source = [], gain = [];
  let currentSourceIdx = 0;
  let buffer = [];
  let bufferCreationTimeMS;
  let lastTimeoutId;

  startNoiseBtn.onclick = () => {
    startNoiseBtn.disabled = true;

    // Simply create all new objects even if it's not the first time click, since
    // we'll need to re-create the source anyway due to the need for new buffer.
    audioCtx = new AudioContext();
    for(let n = 0; n < totalSources; ++n)
    {
      gain[n] = audioCtx.createGain();
      gain[n].connect(audioCtx.destination);
      source[n] = audioCtx.createBufferSource();
      source[n].loop = true;
      source[n].connect(gain[n]);

      buffer[n] = new AudioBuffer({
        numberOfChannels: numChannels,
        length: frameCount,
        sampleRate: audioCtx.sampleRate,
      });
    }

    const time0 = performance.now();
    for(let n = 0; n < totalSources; ++n)
      for(let channel = 0; channel < numChannels; channel++)
        createSoundBuffer(buffer[n].getChannelData(channel));
    const time1 = performance.now();
    bufferCreationTimeMS = (time1-time0) / totalSources;

    gain[0].gain.setValueAtTime(0, audioCtx.currentTime);
    gain[0].gain.linearRampToValueAtTime(1, audioCtx.currentTime + updatePeriod);

    gain[1].gain.setValueAtTime(0, audioCtx.currentTime);
    gain[1].gain.setValueAtTime(0, audioCtx.currentTime + updatePeriod);

    for(let n = 0; n < totalSources; ++n)
    {
      source[n].buffer = buffer[n];
      source[n].start();
    }

    lastTimeoutId = window.setTimeout(updateSound, 1000*updatePeriod - bufferCreationTimeMS);

    stopNoiseBtn.disabled = false;
  };
  function updateSound()
  {
    const origSourceIdx = currentSourceIdx;
    currentSourceIdx = currentSourceIdx == 0 ? 1 : 0;

    for(let channel = 0; channel < numChannels; channel++)
      createSoundBuffer(buffer[currentSourceIdx].getChannelData(channel));
    source[currentSourceIdx].stop();
    source[currentSourceIdx].disconnect(gain[currentSourceIdx]);
    source[currentSourceIdx] = audioCtx.createBufferSource();
    source[currentSourceIdx].loop = true;
    source[currentSourceIdx].connect(gain[currentSourceIdx]);
    source[currentSourceIdx].buffer = buffer[currentSourceIdx];
    source[currentSourceIdx].start();

    // Simple summation of noises with linearly changing mean amplitude results
    // in a non-constant volume of the sum, because the additive quantity here
    // is power, which is quadratic in amplitude. So we need to modulate each
    // amplitude as a square root, so that squaring it will yield a linear change.
    const numAmplitudePoints = 30;
    for(let n = 0; n < numAmplitudePoints; ++n)
    {
      const alpha = n / (numAmplitudePoints - 1);
      const linearAmp0to1 = alpha;
      const linearAmp1to0 = 1 - linearAmp0to1;
      gain[origSourceIdx].gain.linearRampToValueAtTime(Math.sqrt(linearAmp1to0), audioCtx.currentTime + updatePeriod * alpha);
      gain[currentSourceIdx].gain.linearRampToValueAtTime(Math.sqrt(linearAmp0to1), audioCtx.currentTime + updatePeriod * alpha);
    }

    lastTimeoutId = window.setTimeout(updateSound, 1000*updatePeriod - bufferCreationTimeMS);
  }

  let maxDataValue = 0;
  function createSoundBuffer(bufferData)
  {
    const data = new ComplexArray(frameCount).map((value, i, n) => {
      value.real = Math.random() * 2 - 1;
    });
    data.FFT();

    function spectrum(freq) { return 10**(-freq/16000 + 2.65*Math.exp(-freq/1250)); }
    data.map((fft, i, n) => {
      const freq = audioCtx.sampleRate / n * i;
      fft.real *= spectrum(audioCtx.sampleRate/2 - Math.abs(freq - audioCtx.sampleRate/2));
      fft.imag = 0;
    });
    const filtered = data.InvFFT();
    if(!maxDataValue)
    {
      data.forEach((value, i) => {
        if(Math.abs(value.real) > maxDataValue)
          maxDataValue = Math.abs(value.real);
      });
    }

    data.forEach((value, i) => {
      bufferData[i] = value.real / maxDataValue;
    });
  }
  stopNoiseBtn.onclick = () => {
    stopNoiseBtn.disabled = true;
    if(lastTimeoutId)
      window.clearTimeout(lastTimeoutId);
    for(let n = 0; n < totalSources; ++n)
      source[n].stop();
    startNoiseBtn.disabled = false;
  };
}
