# Stress-test and hardening report

This report records the failure modes exercised during the August 2026 hardening pass.
It focuses on predictable failure under malformed input or load rather than UI polish.

## High-impact flaws fixed

1. **Unbounded prediction queue under burst load.** Inference was serialized by a lock,
   but any number of `/predict` requests could still queue around it. The API now admits
   at most `MAX_CONCURRENT_PREDICTIONS` requests (default `2`) and returns `429` with
   `Retry-After` when full, while `/health` remains responsive.
2. **Upload limit could be bypassed until after multipart parsing.** The middleware now
   rejects oversized declared bodies early and also counts streamed/chunked request
   bytes before passing the body to Starlette, so omitting `Content-Length` does not
   bypass the request ceiling.
3. **MIME spoofing / unsupported decodable formats.** Pillow used to accept any image it
   could decode if the client claimed an allowed MIME type. The decoder now verifies the
   real file format is a static JPEG, PNG, or WebP, rejects animation, and normalizes EXIF
   orientation before inference.
4. **Malformed model output could be silently flattened and normalized.** A tensor such
   as `(5, 1)` or positive non-softmax scores could be accepted as five probabilities.
   The service now requires exactly one `(5,)` or `(1, 5)` output, finite values in
   `[0, 1]`, and a sum within floating-point tolerance of `1`.
5. **Non-finite Grad-CAM state could produce misleading rendering.** Feature maps,
   classifier weights, heat maps, and overlay alpha are now validated before rendering;
   explanation failure still degrades safely to `heatmap: null`.
6. **Frontend trusted only two fields of the API response.** It now validates confidence,
   probability ranges/sum, predicted-class consistency, and the heatmap data-URI type and
   size before rendering.
7. **Network calls could hang indefinitely.** Health checks now time out after 5 seconds
   and image analysis after 120 seconds, while user-triggered aborts still cancel cleanly.
8. **Low-confidence contradiction in clinical guidance.** A low-confidence `No DR`/mild
   result could show a warning and simultaneously say “No referral indicated.” The UI
   and exported note now require manual review before clearance when confidence is low.
9. **Duplicate-analysis race.** A ref-level in-flight guard closes the small window before
   React commits `analyzing=true`, and the patient/file metadata are snapshotted at request
   start so edits cannot mismatch the returned screening record.
10. **No automated production-build/model regression gate.** CI now runs frontend tests +
    Vite build and the full backend suite, including the committed real Keras model on
    Linux x64.

## Stress scenarios executed locally

- 2,000 random byte strings through the image decoder: all failed as controlled
  `InvalidImageError` values rather than leaking unexpected exceptions.
- 10,000 valid random five-class probability vectors: all remained finite and normalized.
- 5,000 random tensors with invalid output shapes: all failed closed.
- 1,000 random finite Grad-CAM feature/weight cases: all produced finite `[0, 1]` maps.
- 100 simultaneous prediction requests against a deliberately slow fake model: 2 were
  admitted and 98 were rejected immediately with `429`; `/health` stayed responsive.
- 81 backend non-real-model regression/hardening tests passed locally.
- 12 frontend contract/fail-safe tests passed locally.
- Changed JSX files were parsed with the TypeScript JSX parser to catch syntax errors.

The real-model test and full Vite production build are intentionally delegated to GitHub
Actions because the local isolated runner does not have the repository's binary model or
npm dependency tree mounted. CI installs the pinned dependencies and runs both checks.
