import { Button } from "@chakra-ui/react";
import domtoimage from "dom-to-image-more";
import { BiDownload } from "react-icons/bi";
import useMirage from "../../hooks/useMirage";

interface DownloadResultButtonProps {
  targetRef: HTMLDivElement | null;
}

const DownloadResultButton = ({ targetRef }: DownloadResultButtonProps) => {
  const { startServer, stopServer } = useMirage();

  if (!targetRef) {
    return null;
  }

  return (
    <Button
      id="download-result-button"
      size="sm"
      variant="outline"
      colorScheme="secondary"
      color="secondary.300"
      leftIcon={<BiDownload />}
      px={2}
      onClick={() => {
        // stop mirage interception
        if (stopServer) {
          stopServer();
        }

        // save style before adjusting
        const defaultStyle = { ...targetRef.style };

        // adjust style to make it look better when saving image
        targetRef.style.width = "105%";
        targetRef.style.height = "105%";
        targetRef.style.paddingBottom = "2em";
        targetRef.style.paddingLeft = "2em";
        targetRef.style.paddingRight = "2em";
        targetRef.style.backgroundColor = "white";

        // hide button
        const downloadButton = targetRef.querySelector(
          "#download-result-button"
        ) as HTMLButtonElement;
        downloadButton.style.visibility = "hidden";

        // save as image
        domtoimage.toJpeg(targetRef).then((dataUrl: string) => {
          downloadButton.style.visibility = "visible";
          console.log({ dataUrl });

          const link = document.createElement("a");
          const ts = new Date()
            .toDateString()
            .split(" ")
            .join("-")
            .toLowerCase();
          link.download = `ecg-prediction-${ts}.jpeg`;
          link.href = dataUrl;
          link.click();

          // revert adjusted style
          targetRef.style.width = defaultStyle.width;
          targetRef.style.height = defaultStyle.height;
          targetRef.style.paddingBottom = defaultStyle.paddingBottom;
          targetRef.style.paddingLeft = defaultStyle.paddingLeft;
          targetRef.style.paddingRight = defaultStyle.paddingRight;
          targetRef.style.backgroundColor = defaultStyle.backgroundColor;
          downloadButton.style.visibility = "visible";

          // restart server
          if (startServer) {
            startServer();
          }
        });
      }}
    >
      ดาวน์โหลดผลทำนาย
    </Button>
  );
};

export default DownloadResultButton;
