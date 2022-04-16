import {
  Box,
  Button,
  Flex,
  Heading,
  Icon,
  ListItem,
  Stack,
  Switch,
  Text,
  UnorderedList,
  useDisclosure,
} from "@chakra-ui/react";
import { FaCaretUp } from "react-icons/fa";
import { predictionResult } from "../pages/ecg";

interface PredictionProps {
  predictionResult: predictionResult;
  onClickHowTo: () => void;
}

interface ValueBoxProps {
  title: string;
  labelLt: string;
  labelRt: string;
  value: number;
}
interface PredictionCardProps {
  title: string;
  description: string;
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
    <Flex flex={1} flexDirection="column" alignItems="center" px={8}>
      <Heading size="md" as="h3" textAlign="center" mb={4}>
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

const PredictionCard = ({
  title,
  description,
  labelLt,
  labelRt,
  value,
}: PredictionCardProps) => {
  const leftOffset = (value * 100).toFixed(2);
  const renderValue = value.toFixed(2);
  const renderOnSide = 1 - value >= 0.5 ? "left" : "right";
  const valueLt = renderOnSide === "left" ? renderValue : "0.00";
  const valueRt = renderOnSide === "right" ? renderValue : "0.00";

  return (
    <Flex flex={1} flexDirection="column" alignItems="center" px={8}>
      <Flex justifyContent="space-between">
        <Box>
          <Heading as="h5" fontSize="2xl" color="secondary.400" mb={1}>
            {title}
          </Heading>
          <Text fontSize="sm">{description}</Text>
        </Box>
        <Box textAlign="center">
          <Text fontSize="sm" mb={3}>
            ระดับความเสี่ยง
          </Text>
          <Heading as="h5" fontSize="2xl" color="primary.300" mb={1}>
            ต่ำ
          </Heading>
        </Box>
      </Flex>
      {/* <Heading size="md" as="h3" textAlign="center" mb={4}>
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
      </Stack> */}
    </Flex>
  );
};

const Prediction = ({ predictionResult, onClickHowTo }: PredictionProps) => {
  const {
    isOpen: isOpenPanel,
    onClose: onClosePanel,
    onOpen: onOpenPanel,
  } = useDisclosure({ defaultIsOpen: true });

  if (!Array.isArray(predictionResult)) {
    return null;
  }

  const mock = predictionResult[0];

  return (
    <Stack direction="column" gap={4} mt={10}>
      <Heading as="h4" fontSize="2xl" color="secondary.400" mb={6}>
        ผลทำนาย
      </Heading>
      <PredictionCard
        title="Myocardial scar"
        description="ความเสี่ยงที่จะมีรอยแผลเป็นที่กล้ามเนื้อหัวใจ"
        labelLt={mock.labelLt}
        labelRt={mock.labelRt}
        value={mock.score}
      />
      {/* <Flex justifyContent="flex-end" alignItems="center">
        <Box>
          <Switch
            defaultChecked
            size="md"
            colorScheme="primary"
            onChange={(e) =>
              e.currentTarget.checked ? onOpenPanel() : onClosePanel()
            }
          />
          &nbsp; เปิด / ปิด การทำนายผล
        </Box>
      </Flex>
      {isOpenPanel && (
        <>
          <Stack
            direction={{ base: "column", md: "row" }}
            gap={{ base: 8, md: 2 }}
          >
            {predictionResult.map(
              ({ prediction_title, score, labelLt, labelRt }) => (
                <ValueBox
                  key={`${prediction_title}-${score}`}
                  title={prediction_title}
                  labelLt={labelLt}
                  labelRt={labelRt}
                  value={score}
                />
              )
            )}
          </Stack>
          <Box textAlign="left" fontSize="sm">
            <UnorderedList>
              <ListItem>
                ความเสี่ยงที่จะมีรอยแผลเป็นที่กล้อมเนื้อหัวใจ (Myocardial scar,
                scar)
              </ListItem>
              <ListItem>
                ค่าประสิทธิภาพการทำงานของหัวใจห้องล่างซ้าย (Left ventricular
                ejection fraction, LVEF)
              </ListItem>
              <ListItem>
                <Button
                  colorScheme="pink"
                  variant="link"
                  onClick={onClickHowTo}
                  size="sm"
                >
                  ดูวิธีการสร้างโมเดล
                </Button>
              </ListItem>
            </UnorderedList>
          </Box>
        </>
      )} */}
    </Stack>
  );
};

export default Prediction;
