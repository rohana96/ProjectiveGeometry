from src.rectification import affine_rect, metric_rect_from_affine_rectified, metric_rect_from_proj
from src.homography import homography_interactive
import argparse


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', help='run type', required=True)
    parser.add_argument('--imgs', nargs='+', help='image names', required=True)
    parser.add_argument('--template', help='template for homography', required=False)
    parser.add_argument('--method', help='method')
    parser.add_argument('--outname', help='output file name; needed for running homography', required=False)
    parser.add_argument('--compute_angles', action='store_true', default=False, help="compute test angles")
    parser.add_argument('--force_annotate', action='store_true', default=False, help="prevent use of cached annotations")
    return parser


def run(args):
    if args.type == 'rectification':
        if args.method == 'affine':
            for img in args.imgs:
                affine_rect(
                    imagename=img,
                    compute_angles=args.compute_angles,
                    force_annotate=args.force_annotate
                )

        elif args.method == 'affine_to_metric':
            for img in args.imgs:
                metric_rect_from_affine_rectified(
                    imagename=img,
                    compute_angles=args.compute_angles,
                    force_annotate=args.force_annotate
                )

        elif args.method == 'direct_metric':
            for img in args.imgs:
                metric_rect_from_proj(
                    imagename=img,
                    compute_angles=args.compute_angles,
                    force_annotate=args.force_annotate
                )

    elif args.type == 'homography':
        template = f'data/homography/{args.template}.jpg'
        for img in args.imgs:
            img = f'data/homography/{img}.jpg'
            homography_interactive(
                imagepath1=img,
                imagepath2=template,
                outname=f'{args.outname}.jpg'
            )
            template = f'out/homography/{args.outname}.jpg'


if __name__ == "__main__":
    parser = argparser()
    args = parser.parse_args()
    run(args)
