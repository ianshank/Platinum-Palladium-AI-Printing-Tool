/**
 * The digital-negative export call must match the multipart contract the
 * server declares (plan item ARC-07).
 *
 * The endpoint reads Form fields, not a JSON body, and streams a file back.
 * The last time a client sent the wrong shape for a Form endpoint, every
 * `.quad` parse failed with "Field required" and nothing caught it, because
 * the client tests mocked the client itself. These assert the request the
 * client actually builds.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { AxiosProgressEvent, AxiosRequestConfig } from 'axios';

const { mockRequest, sent } = vi.hoisted(() => {
  const sent: AxiosRequestConfig[] = [];
  return {
    sent,
    // Capture the config here rather than reading mock.calls, whose element
    // type is `any` and trips the no-unsafe-assignment rule.
    mockRequest: vi.fn((config: AxiosRequestConfig) => {
      sent.push(config);
      return Promise.resolve({ data: new Blob() });
    }),
  };
});

// client.ts only ever calls axios.create(), so the mock needs nothing else.
vi.mock('axios', () => ({
  default: {
    create: () => ({
      request: mockRequest,
      interceptors: {
        request: { use: vi.fn() },
        response: { use: vi.fn() },
      },
    }),
  },
}));

const { api } = await import('./client');

function sentConfig(): AxiosRequestConfig {
  expect(sent).toHaveLength(1);
  return sent[0]!;
}

function sentForm(): FormData {
  // AxiosRequestConfig.data is `any`; narrow it before touching it.
  const data: unknown = sentConfig().data;
  expect(data).toBeInstanceOf(FormData);
  return data as FormData;
}

function pngFile(): File {
  return new File([new Uint8Array([137, 80, 78, 71])], 'scan.png', {
    type: 'image/png',
  });
}

describe('api.negative.export', () => {
  beforeEach(() => {
    sent.length = 0;
    mockRequest.mockClear();
  });

  it('posts multipart form data and asks for a blob back', async () => {
    await api.negative.export({ file: pngFile() });

    const config = sentConfig();
    expect(config.method).toBe('POST');
    expect(config.url).toBe('/api/export/negative');
    expect(config.responseType).toBe('blob');
    expect(config.headers?.['Content-Type']).toBe('multipart/form-data');
  });

  it('defaults to a 16-bit grayscale negative', async () => {
    await api.negative.export({ file: pngFile() });

    const form = sentForm();
    expect(form.get('format')).toBe('tiff_16bit');
    expect(form.get('color_mode')).toBe('grayscale');
    expect(form.get('invert')).toBe('true');
    expect(form.get('name')).toBe('negative');
  });

  it('sends a stored curve by id', async () => {
    await api.negative.export({ file: pngFile(), curveId: 'abc-123' });

    expect(sentForm().get('curve_id')).toBe('abc-123');
  });

  it('sends densities as repeated fields, which is what Form(list) reads', async () => {
    await api.negative.export({ file: pngFile(), densities: [0.1, 0.5, 1.2] });

    expect(sentForm().getAll('densities')).toEqual(['0.1', '0.5', '1.2']);
  });

  it('omits the curve entirely when neither source is given', async () => {
    await api.negative.export({ file: pngFile() });

    const form = sentForm();
    expect(form.get('curve_id')).toBeNull();
    expect(form.getAll('densities')).toEqual([]);
  });

  it('honours an explicit format, mode and inversion', async () => {
    await api.negative.export({
      file: pngFile(),
      format: 'png',
      colorMode: 'preserve',
      invert: false,
      name: 'my print',
    });

    const form = sentForm();
    expect(form.get('format')).toBe('png');
    expect(form.get('color_mode')).toBe('preserve');
    expect(form.get('invert')).toBe('false');
    expect(form.get('name')).toBe('my print');
  });

  it('reports upload progress when a callback is supplied', async () => {
    const onProgress = vi.fn();

    await api.negative.export({ file: pngFile(), onProgress });

    const handler = sentConfig().onUploadProgress;
    expect(handler).toBeTypeOf('function');
    handler?.({ loaded: 50, total: 200, bytes: 50 } as AxiosProgressEvent);
    expect(onProgress).toHaveBeenCalledWith(25);
  });
});
