import {
  Box,
  Button,
  CloseButton,
  Flex,
  Image,
  Text,
  VStack,
} from "@chakra-ui/react";
import React, { useEffect, useState } from "react";
import { useDropzone } from "react-dropzone";

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

  useEffect(() => {
    // Make sure to revoke the data uris to avoid memory leaks
    files.forEach((file) => URL.revokeObjectURL(file.preview));
  }, [files]);

  return (
    <VStack w="100%" gap={4}>
      {files.length <= 0 && (
        <Flex
          {...getRootProps({ className: "dropzone" })}
          justifyContent={"center"}
          alignItems={"center"}
          w="100%"
          background={isDragAccept ? "pink.50" : "gray.50"}
          border={`2px solid ${isDragReject ? "red" : "pink"}`}
          borderRadius="md"
          borderStyle="dashed"
          py={12}
        >
          <input {...getInputProps()} />
          <Text color="gray.500">
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

      {files.length > 0 && (
        <>
          <Box
            border="2px solid pink"
            borderRadius="md"
            p={1}
            borderStyle="dashed"
            position="relative"
          >
            <Text
              textAlign={"center"}
              my={2}
              fontSize={14}
              color="gray.600"
              fontStyle="italic"
            >
              {files[0].path}
            </Text>
            <Image src={files[0].preview} alt="preview image" />
            <CloseButton
              onClick={() => {
                setFiles([]);
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
