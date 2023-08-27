#include <linux/types.h>

#include "arm_neon.h"

extern int g_param_mono_cutoff;
extern int g_param_mono_invert;

static const uint8x8_t bitmask = {128, 64, 32, 16, 8, 4, 2, 1};
size_t simd_impl(u8 *buf, int width, int local_width, int height)  {
	int const tagged_line_len = 2 + width / 8;
	u8 const invert = g_param_mono_invert ? 0xFF : 0x00;
	u8 const cutoff = g_param_mono_cutoff;

	uint8x8_t cutoff_vec = vld1_dup_u8(&cutoff);
	uint8x8_t invert_mask = vld1_dup_u8(&invert);

	// Iterate over lines from [0, height)
	for (int line = 0; line < height; line++) {

		// Iterate over chunks of 8 source grayscale bytes
		// Each 8-byte source chunk will map to one destination mono byte
		for(int chunk = 0; chunk < local_width; chunk+=8) {
			uint8x8_t v1 = vld1_u8(buf + (line*width) + (chunk+0)*8);
			uint8x8_t v2 = vld1_u8(buf + (line*width) + (chunk+1)*8);
			uint8x8_t v3 = vld1_u8(buf + (line*width) + (chunk+2)*8);
			uint8x8_t v4 = vld1_u8(buf + (line*width) + (chunk+3)*8);
			uint8x8_t v5 = vld1_u8(buf + (line*width) + (chunk+4)*8);
			uint8x8_t v6 = vld1_u8(buf + (line*width) + (chunk+5)*8);
			uint8x8_t v7 = vld1_u8(buf + (line*width) + (chunk+6)*8);
			uint8x8_t v8 = vld1_u8(buf + (line*width) + (chunk+7)*8);

			v1 = vcge_u8(v1, cutoff_vec);
			v2 = vcge_u8(v2, cutoff_vec);
			v3 = vcge_u8(v3, cutoff_vec);
			v4 = vcge_u8(v4, cutoff_vec);
			v5 = vcge_u8(v5, cutoff_vec);
			v6 = vcge_u8(v6, cutoff_vec);
			v7 = vcge_u8(v7, cutoff_vec);
			v8 = vcge_u8(v8, cutoff_vec);

			v1 &= bitmask;
			v2 &= bitmask;
			v3 &= bitmask;
			v4 &= bitmask;
			v5 &= bitmask;
			v6 &= bitmask;
			v7 &= bitmask;
			v8 &= bitmask;

			v1 = vpadd_u8(v1, v2);
			v3 = vpadd_u8(v3, v4);
			v5 = vpadd_u8(v5, v6);
			v7 = vpadd_u8(v7, v8);

			v1 = vpadd_u8(v1, v3);
			v5 = vpadd_u8(v5, v7);

			v1 = vpadd_u8(v1, v5);
			v1 ^= invert_mask;
			
			vst1_u8(buf + (line * tagged_line_len) + 1 + chunk, v1);
		}
	}

	return height * tagged_line_len;
}
