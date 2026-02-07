import { useMutation, useQuery } from '@tanstack/react-query';
import { api } from '../client';

export function useGenerateCurve() {
    return useMutation({
        mutationFn: (data: { measurements: number[]; curve_type?: string; name?: string }) =>
            // Map curve_type to type to match client API
            api.curves.generate({ ...data, type: data.curve_type }),
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
