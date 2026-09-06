#include <cstdlib>                       /* binVIO addition */
#ifdef BINVIO_FRONTEND_AVAILABLE          /* binVIO addition */
#include "binvio/ingest/gray_sequence.hpp"
#endif
#include "video_input.hpp"
#include "../util/allocator.hpp"
#include "../util/logging.hpp"
#include "../odometry/parameters.hpp"
#include "videoutil.hpp"
#include "../util/bounded_processing_queue.hpp"

#include <cassert>
#include <iostream>
#include <sstream>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

/* PCL addition */
#include "../util/global_seal.hpp"

namespace {

struct Reader {
    virtual ~Reader() = default;
    virtual bool read(cv::Mat &target) = 0;
    virtual bool isOk() const = 0;
};

class VideoInputImplementation : public VideoInput {
private:
    std::string videoPath;
    std::unique_ptr<Reader> reader;
    cv::Mat resizeSource, colorSource;
    int resizeWidth = -1, resizeHeight = -1;

    static constexpr size_t BUFFER_SIZE = 4;
    util::BoundedInputQueue<cv::Mat> queue;
    cv::Mat constant_frame = cv::Mat(cv::Size(752, 480), CV_8UC1);

public:
    VideoInputImplementation(const std::string &videoPath, std::unique_ptr<Reader> reader, bool ownThread, bool convertToGray) :
        videoPath(videoPath),
        reader(std::move(reader)),
        queue(ownThread ? BUFFER_SIZE : 0, [this, convertToGray](cv::Mat &frame) -> bool {
            /* PCL addition: Avoid loading the frames, HybVIO will not be needing them. */
            if (SEAL_PEmu->getMode() == SEALPowerEmulator::Mode::LOAD) {
                frame = constant_frame;
                return true;
            }
            const bool resized = resizeWidth > 0;
            cv::Mat &target = convertToGray ? colorSource : (resized ? resizeSource : frame);
            if (this->reader->read(target)) {
                if (convertToGray) {
                    cv::Mat &convertTarget = resized ? resizeSource : frame;
                    switch (target.channels()) {
                        case 1: target.copyTo(convertTarget); break;
                        case 3: cv::cvtColor(target, convertTarget, cv::COLOR_BGR2GRAY); break;
                        case 4: cv::cvtColor(target, convertTarget, cv::COLOR_BGRA2GRAY); break;
                        default: assert(false && "invalid color format"); break;
                    }
                }
                if (resized) {
                    cv::resize(resizeSource, frame, cv::Size(resizeWidth, resizeHeight), 0, 0, cv::INTER_CUBIC);
                }

                /* PCL addition:
                 * If TemporalProcessor is on, and denoiser is on, it will apply the median filter.
                 * Same for the edge detection.
                 */
                SEAL->temporal_process(frame);
            } else {
                return false;
            }
            return true;
        })
    {}

    double probeFPS() final {
        return videoutil::ffprobeFps(videoPath);
    }

    void probeResolution(int &w, int &h) final {
        bool success = videoutil::ffprobeResolution(videoPath, w, h);
        assert(success && w > 0 && h > 0);
    }

    std::shared_ptr<cv::Mat> readFrame() final {
        return queue.get();
    }

    void resize(int width, int height) final {
        resizeWidth = width;
        resizeHeight = height;
    }
};

struct FFMpegReader: Reader {
    int width = 0;
    int height = 0;
    FILE *pipe = nullptr;

    FFMpegReader(const std::string &videoPath, const std::string vf) {
        bool success = videoutil::ffprobeResolution(videoPath, width, height);
        if (success) {
            assert(success && width > 0 && height > 0);
            std::stringstream filters;
            if (!vf.empty()) {
                filters << " -vf \"" << vf << "\"";
            }

            constexpr bool VERBOSE = false;

            std::stringstream ss; ss << "ffmpeg -i "
                << videoPath
                << " -f rawvideo -vcodec rawvideo -vsync vfr"
                << filters.str()
                << " -pix_fmt bgr24 -"
                << (VERBOSE ? "" : " 2>/dev/null");

            log_debug("Running: %s", ss.str().c_str());
            pipe = popen(ss.str().c_str(), "r");
        }
    }

    ~FFMpegReader() {
        fflush(pipe);
        pclose(pipe);
    }

    bool read(cv::Mat &frame) final {

        if (frame.empty()) {
            frame = cv::Mat(height, width, CV_8UC3);
        }
        assert(frame.rows == height && frame.cols == width);
        assert(frame.type() == CV_8UC3);
        int n = height * width * 3;
        assert(pipe);
        assert(frame.data);
        int count = fread(frame.data, 1, n, pipe);

        return count == n;
    }

    bool isOk() const final {
        return pipe != nullptr;
    }
};

#ifdef BINVIO_FRONTEND_AVAILABLE
/* binVIO addition: frames from a raw grayscale sequence, with NO DECODER.
 *
 * WHY THIS EXISTS. HybVIO has exactly two readers and both take a video file:
 * FFMpegReader pipes from the ffmpeg binary and OpenCVReader uses
 * cv::VideoCapture. Both decode to **BGR24** and the caller then converts back
 * to gray -- on EuRoC, which is grayscale to begin with. The .mp4 exists only
 * because SEAL's download_euroc.py encoded EuRoC's PNG sequence into one; a
 * camera does not produce H.264.
 *
 * The cost is not small and it is paid by BOTH frontends, which is what makes it
 * worth removing rather than merely noting: libavcodec is 23.5% of a
 * bincv-seal-binary run (TR-02) on frames that path discards entirely, and the
 * decoder's queue is 9.75 MB (M-19). A cost shared by numerator and denominator
 * COMPRESSES the ratio, so measuring with it in place understates what the
 * frontend replacement is worth in a setting where frames come from a sensor.
 *
 * This reads binVIO's .graysq -- an mmap'd 8-bit sequence, the same storage idea
 * binvio-mobile-raw already uses (D-9) -- and hands out CV_8UC1 directly. It is
 * deliberately a *Reader* and not a whole VideoInput, so the queue, the
 * conversion path, the timestamps and the frame loop are all byte-for-byte what
 * the video path does. **One variable changes: where the pixels come from.**
 *
 * Selected by BINVIO_RAW_FRAMES=<file.graysq>, so it is off unless asked for and
 * is available to the BASELINE as well as to binVIO -- a fair comparison needs
 * both sides on it. */
struct RawGrayReader : Reader {
    binvio::GraySequenceReader seq;
    size_t next = 0;
    bool ok = false;

    explicit RawGrayReader(const std::string &path) {
        ok = seq.open(path);
        if (!ok) {
            std::cout << "binVIO: " << seq.error() << std::endl;
        } else {
            log_debug("binVIO: raw frames from %s (%u frames, %ux%u)", path.c_str(),
                      seq.frames(), seq.width(), seq.height());
        }
    }

    bool read(cv::Mat &frame) final {
        const uint8_t *p = seq.frame(next);
        if (!p) return false;   /* end of sequence, as a short video would */
        ++next;
        /* Wraps the mapping rather than copying it, then copies once into the
         * caller's buffer -- which is what a VideoCapture read does too, so the
         * two paths differ in the decode and in nothing else. */
        const cv::Mat view(static_cast<int>(seq.height()), static_cast<int>(seq.width()),
                           CV_8UC1, const_cast<uint8_t *>(p),
                           static_cast<size_t>(seq.stride()));
        view.copyTo(frame);
        return true;
    }

    bool isOk() const final { return ok; }
};
#endif  /* BINVIO_FRONTEND_AVAILABLE */

struct OpenCVReader : Reader {
    cv::VideoCapture videoCapture;

    OpenCVReader(const std::string &videoPath) {
        videoCapture = cv::VideoCapture(videoPath);
    }

    bool read(cv::Mat &frame) final {
        return videoCapture.read(frame);
    }

    bool isOk() const final {
        return videoCapture.isOpened();
    }
};
}

std::unique_ptr<VideoInput> VideoInput::build(
        const std::string &fileName,
        const bool convertVideoToGray,
        const bool videoReaderThreads,
        const bool ffmpeg,
        const std::string &vf) {
    /* binVIO addition: the raw-frame path wins over both video readers when it is
     * asked for, because it is asked for by naming a file. */
#ifdef BINVIO_FRONTEND_AVAILABLE
    const char *rawFrames = std::getenv("BINVIO_RAW_FRAMES");
#else
    const char *rawFrames = nullptr;
#endif
    auto reader = rawFrames
#ifdef BINVIO_FRONTEND_AVAILABLE
        ? std::unique_ptr<Reader>(new RawGrayReader(rawFrames))
#else
        ? std::unique_ptr<Reader>(nullptr)
#endif
        : (ffmpeg
            ? std::unique_ptr<Reader>(new FFMpegReader(fileName, vf))
            : std::unique_ptr<Reader>(new OpenCVReader(fileName)));
    if (!reader->isOk()) {
        std::cout << "Couldn't open video " << fileName << std::endl;
        return nullptr;
    }
    return std::unique_ptr<VideoInput>(new VideoInputImplementation(fileName,
        std::move(reader),
        videoReaderThreads,
        convertVideoToGray));
}

VideoInput::~VideoInput() = default;
