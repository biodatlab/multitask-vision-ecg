import {
  Box,
  Button,
  Flex,
  Heading,
  HStack,
  Icon,
  Image,
  Text,
  VStack,
} from "@chakra-ui/react";
import React, { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { BiImage, BiUpload } from "react-icons/bi";

interface FileWithPreview extends File {
  preview: string;
}

interface DropzoneProp {
  onClearFile: () => void;
  onSubmit: (file: FileWithPreview) => void;
  isLoading: boolean;
}

const Dropzone = ({ onClearFile, onSubmit, isLoading }: DropzoneProp) => {
  const [files, setFiles] = useState([] as any);

  const {
    acceptedFiles,
    getRootProps,
    getInputProps,
    isDragAccept,
    isDragReject,
  } = useDropzone({
    accept: "image/jpg,image/jpeg,image/png,application/pdf",
    maxFiles: 1,
    onDrop: (acceptedFiles) => {
      setFiles(
        acceptedFiles.map((file) =>
          Object.assign(file, {
            preview: URL.createObjectURL(file),
          })
        )
      );
    },
  });

  const handleClearFile = useCallback(() => {
    // revoke the data uris to avoid memory leaks
    files.forEach((file: FileWithPreview) => URL.revokeObjectURL(file.preview));
    // remove files
    setFiles([]);
    // remove result
    onClearFile();
  }, [files, onClearFile]);

  const selectedFile = files?.[0];
  const isPdfFile = selectedFile?.type === "application/pdf";

  return (
    <VStack w="100%" gap={4}>
      {/* no selected file */}
      {!selectedFile && (
        <Box
          w={["100%", "md"]}
          background={isDragAccept ? "secondary.50" : "white"}
          borderRadius="xl"
          boxShadow="lg"
          p={6}
          {...getRootProps({ className: "dropzone" })}
        >
          <Flex
            justifyContent={"center"}
            alignItems={"center"}
            borderWidth={1}
            borderStyle="dashed"
            borderColor={isDragReject ? "red" : "secondary.300"}
            borderRadius="xl"
            py={8}
          >
            <input {...getInputProps()} />
            <Box color="gray.500">
              <Icon as={BiImage} fontSize={54} color="secondary.400" mb={2} />

              <Box maxW={["80%", "75%"]} mx="auto">
                <Heading
                  as="h4"
                  fontSize={[18, 21]}
                  fontWeight="regular"
                  lineHeight="tall"
                  color="secondary.400"
                  mb={3}
                >
                  ลากภาพสแกน ECG มาบริเวณนี้เพื่ออัปโหลด หรือ
                </Heading>
              </Box>

              <Box mb={4}>
                <Button
                  variant="outline"
                  colorScheme="secondary"
                  color="secondary.400"
                  leftIcon={<Icon as={BiUpload} color="red.300" />}
                  px={3}
                >
                  อัปโหลดจากเครื่อง
                </Button>
              </Box>

              <Box maxW={["80%", "55%"]} mx="auto">
                <Text
                  as="span"
                  fontSize="xs"
                  textAlign="center"
                  color={isDragReject ? "red" : undefined}
                  fontWeight={isDragReject ? "bold" : undefined}
                >
                  รองรับไฟล์ประเภท JPG, JPEG, PNG และ PDF รองรับได้ครั้งละ 1
                  ไฟล์เท่านั้น
                </Text>
              </Box>
            </Box>
          </Flex>
        </Box>
      )}

      {/* file selected */}
      {selectedFile && (
        <Box>
          <Box
            w={{ base: "100%", md: "2xl", xl: "4xl" }}
            backgroundColor="white"
            borderRadius="lg"
            boxShadow="lg"
            mb={8}
          >
            {/* previewer */}
            <Box
              position="relative"
              w={isPdfFile ? "100%" : "auto"}
              borderWidth={1}
              borderStyle="dashed"
              borderColor="primary.300"
              borderRadius="xl"
              overflow="hidden"
              p={1}
            >
              <Text
                textAlign={"center"}
                my={2}
                fontSize={14}
                fontStyle="italic"
              >
                {files[0].path}
              </Text>
              {isPdfFile ? (
                <embed
                  src={files[0].preview}
                  style={{ width: "100%", height: "50vh" }}
                />
              ) : (
                <Image src={files[0].preview} alt="preview image" />
              )}
            </Box>
          </Box>

          {/* buttons */}
          <Flex justifyContent="flex-end">
            <HStack>
              <Button
                variant="outline"
                colorScheme="secondary"
                color="secondary.400"
                px={3}
                onClick={() => {
                  handleClearFile();
                }}
                visibility={isLoading ? "hidden" : "visible"}
              >
                ลบภาพ
              </Button>
              <Button
                colorScheme="primary"
                backgroundColor="primary.300"
                px={3}
                onClick={() => onSubmit(selectedFile)}
                isLoading={isLoading}
              >
                ทำนายผล
              </Button>
            </HStack>
          </Flex>
        </Box>
      )}
    </VStack>
  );
};

export default Dropzone;
