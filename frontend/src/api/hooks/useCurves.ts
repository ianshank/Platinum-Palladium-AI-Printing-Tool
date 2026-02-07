import { useMutation, useQuery } from '@tanstack/react-query';
import { api } from '../client';

export function useGenerateCurve() {
    return useMutation({
        mutationFn: (data: { measurements: number[]; curve_type?: string; name?: string }) =>
            // Map curve_type to type to match client API
            api.curves.generate({ measurements: data.measurements, ...(data.curve_type ? { type: data.curve_type } : {}), ...(data.name ? { name: data.name } : {}) }),
    });
}

export function useGetCurve(id: string) {
    return useQuery({
        queryKey: ['curve', id],
        queryFn: () => api.curves.get(id),
        enabled: !!id,
    });
}

export function useExportCurve() {
    return useMutation({
        mutationFn: (data: { curveId: string; format: string }) =>
            api.curves.export(data),
    });
}
