import {
  Box,
  Flex,
  Heading,
  Icon,
  Stack,
  Switch,
  Text,
  useDisclosure,
} from "@chakra-ui/react";
import { FaCaretUp } from "react-icons/fa";

interface PredictionProps {
  normality: number;
  lvefgteq40: number;
  lveflw50: number;
}

interface ValueBoxProps {
  title: string;
  labelLt: string;
  labelRt: string;
  value: number;
}

const ValueBox = ({ title, labelLt, labelRt, value }: ValueBoxProps) => {
  const leftOffset = (value * 100).toFixed(2);
  const renderValue = value.toFixed(2);
  const renderOnSide = 1 - value >= 0.5 ? "left" : "right";
  const valueLt = renderOnSide === "left" ? renderValue : "0.00";
  const valueRt = renderOnSide === "right" ? renderValue : "0.00";

  return (
    <Flex flex={1} flexDirection="column" alignItems="center" px={8} mb={8}>
      <Heading size="md" as="h3" color="gray.600" textAlign="center" mb={4}>
        {title}
      </Heading>
      <Stack
        direction="row"
        w="100%"
        gap={{ base: 0, md: 1 }}
        alignItems="flex-start"
        justifyContent="center"
      >
        <Box>
          <Text
            fontSize="xs"
            textAlign="center"
            color="gray.500"
            fontWeight={renderOnSide === "left" ? "bold" : "normal"}
          >
            <Text
              as="span"
              fontSize="125%"
              opacity={renderOnSide === "left" ? 1 : 0}
            >
              {valueLt}
            </Text>
            <br />
            {labelLt}
          </Text>
        </Box>
        <Flex maxW="65%" flex={1} pt={1} position="relative">
          <Box
            h={4}
            w="100%"
            borderRadius="xl"
            bgGradient="linear(to-r, pink.100, pink.200, red.400, red.500)"
          />
          <Box
            h={4}
            w="3px"
            position="absolute"
            top={1}
            left={`calc(${leftOffset}% - 1.5px)`}
            backgroundColor="gray.600"
            borderRadius="md"
          />
          <Box position="absolute" top={4} left={`calc(${leftOffset}% - 10px)`}>
            <Icon w="20px" h="20px" as={FaCaretUp} color="gray.600" />
          </Box>
        </Flex>
        <Box>
          <Text
            fontSize="xs"
            textAlign="center"
            color={renderOnSide === "right" ? "red.500" : "gray.500"}
            fontWeight={renderOnSide === "right" ? "bold" : "normal"}
          >
            <Text
              as="span"
              fontSize="125%"
              opacity={renderOnSide === "right" ? 1 : 0}
            >
              {valueRt}
            </Text>
            <br />
            {labelRt}
          </Text>
        </Box>
      </Stack>
    </Flex>
  );
};

const Prediction = ({ normality, lvefgteq40, lveflw50 }: PredictionProps) => {
  const {
    isOpen: isOpenPanel,
    onClose: onClosePanel,
    onOpen: onOpenPanel,
  } = useDisclosure();

  return (
    <Stack direction="column" gap={4}>
      <Flex justifyContent="flex-end" alignItems="center">
        <Box>
          <Switch
            size="md"
            colorScheme="pink"
            onChange={(e) =>
              e.currentTarget.checked ? onOpenPanel() : onClosePanel()
            }
          />
          &nbsp; เปิด / ปิด ความน่าจะเป็น
        </Box>
      </Flex>
      {isOpenPanel && (
        <Stack
          direction={{ base: "column", md: "row" }}
          gap={{ base: 8, md: 2 }}
        >
          <ValueBox
            title="แผลเป็น"
            labelLt="ไม่มี"
            labelRt="มี"
            value={normality}
          />
          <ValueBox
            title="LVEF < 40"
            labelLt=">= 40"
            labelRt="< 40"
            value={lvefgteq40}
          />
          <ValueBox
            title="LVEF < 50"
            labelLt=">= 50"
            labelRt="< 50"
            value={lveflw50}
          />
        </Stack>
      )}
    </Stack>
  );
};

export default Prediction;
