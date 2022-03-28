import {
  Box,
  Button,
  CloseButton,
  Flex,
  Icon,
  Image,
  Text,
  VStack,
} from "@chakra-ui/react";
import React, { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { FaCloudUploadAlt } from "react-icons/fa";

const Dropzone = () => {
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
    files.forEach((file: any) => URL.revokeObjectURL(file.preview));
    // remove files
    setFiles([]);
  }, [files]);

  const previewFile = files?.[0];
  const isPdfFile = previewFile?.type === "application/pdf";

  return (
    <VStack w="100%" gap={4}>
      {!previewFile && (
        <Flex
          {...getRootProps({ className: "dropzone" })}
          justifyContent={"center"}
          alignItems={"center"}
          w="100%"
          background={isDragAccept ? "pink.50" : "gray.50"}
          border={`2px solid ${isDragReject ? "red" : "pink"}`}
          borderRadius="md"
          borderStyle="dashed"
          py={6}
        >
          <input {...getInputProps()} />
          <Text color="gray.500">
            <Icon as={FaCloudUploadAlt} w={14} h={14} />
            <br />
            ลากไฟล์มาที่นี่หรือคลิกที่นี่เพื่อเลือกไฟล์ &nbsp;
            <Text
              as="span"
              color={isDragReject ? "red" : undefined}
              fontWeight={isDragReject ? "bold" : undefined}
            >
              รองรับไฟล์ JPG, JPEG, PNG และ PDF ครั้งละ 1 ไฟล์เท่านั้น
            </Text>
          </Text>
        </Flex>
      )}

      {previewFile && (
        <>
          <Box
            border="2px solid pink"
            borderRadius="md"
            p={1}
            borderStyle="dashed"
            position="relative"
            w={isPdfFile ? "100%" : "auto"}
          >
            <Text textAlign={"center"} my={2} fontSize={14} fontStyle="italic">
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
            <CloseButton
              onClick={() => {
                handleClearFile();
              }}
              size="md"
              position="absolute"
              top={1}
              right={1}
            />
          </Box>
          <Button colorScheme={"pink"}>ทำนาย</Button>
        </>
      )}
    </VStack>
  );
};

export default Dropzone;
