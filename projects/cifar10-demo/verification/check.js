(async () => {
  const report = document.getElementById('verification-result');
  try {
    const fixtures = await fetch('verification/fixtures.json').then(r => r.json());
    const deadline = Date.now() + 30000;
    while (!state.session) {
      if (Date.now() > deadline) throw new Error('等待模型加载超时');
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    const results = [];
    for (const fixture of fixtures) {
      const image = await loadImage(fixture.file);
      drawModelInput(image);
      const input = imageToTensor();
      let inputError = 0;
      const mean = [0.4914, 0.4822, 0.4465];
      const std = [0.2470, 0.2435, 0.2616];
      for (let p = 0; p < 1024; p++) {
        for (let c = 0; c < 3; c++) {
          const expected = Math.fround((fixture.rgb[p * 3 + c] / 255 - mean[c]) / std[c]);
          inputError = Math.max(inputError, Math.abs(input.data[c * 1024 + p] - expected));
        }
      }
      const output = await state.session.run({ images: input });
      const logits = Array.from(output.logits.data);
      const maxAbsError = Math.max(...logits.map((v, i) => Math.abs(v - fixture.logits[i])));
      const predicted = logits.indexOf(Math.max(...logits));
      const referencePredicted = fixture.logits.indexOf(Math.max(...fixture.logits));
      if (inputError !== 0 || maxAbsError > 0.0003 || predicted !== referencePredicted || predicted !== fixture.expectedClass) {
        throw new Error(JSON.stringify({file: fixture.file, inputError, maxAbsError, predicted, referencePredicted}));
      }
      results.push({file: fixture.file, inputError, maxAbsError, predicted});
    }
    report.textContent = JSON.stringify({passed: true, modelVersion: state.metadata.modelVersion, results}, null, 2);
  } catch (error) {
    report.textContent = JSON.stringify({passed: false, error: String(error)}, null, 2);
  }
})();
